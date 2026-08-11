# BLE WiFi provisioning

Provision a brand-new Reachy Mini's WiFi from a phone **over Bluetooth only** —
the user never has to join the robot's WiFi hotspot.

The phone talks to the GATT command service exposed by `bluetooth_service.py`
(system Python, stdlib-only). That service **proxies** to the daemon's
`/wifi/*` FastAPI routes on `127.0.0.1:8000`; the daemon owns all `nmcli`
plumbing **and** all cryptography (it has the `cryptography` lib in its venv).

## GATT layout

| | UUID | Properties |
|---|---|---|
| Service | `12345678-…-56789abcdef0` | — |
| Command | `12345678-…-56789abcdef1` | `write` |
| Response | `12345678-…-56789abcdef2` | `read`, `notify` |

Write UTF-8 command strings to **Command**. Subscribe to **Response**
notifications for results.

## Async response model

WiFi commands proxy to the daemon and can block for ~10 s (`nmcli rescan`), so
they run **off** the BLE mainloop. A write returns an immediate
`OK: working` ack on the Response characteristic; the **real result arrives as
a later notification**. So the client MUST:

1. `StartNotify` on the Response characteristic.
2. Write the command.
3. Await the next Response notification for the actual payload.

(`PING`, `PIN_…`, `WIFI_KEYEX` reply synchronously — the Response value is set
before the write returns. They also re-notify, so awaiting the notification
works uniformly.)

## Auth / session

`PIN_<pin>` opens a **TTL-bounded session** (default 300 s). The PIN is the
last 5 chars of the device serial (printed on the robot). Within the window a
client can chain scan → connect → status without re-authing; after it expires,
privileged commands are refused again (write `PIN_…` to renew).

| Command | Auth | Result (via notification) |
|---|---|---|
| `WIFI_STATUS` | public | `{"mode","connected","error"}`, plus `"known":[…]` **only when authed** |
| `WIFI_KEYEX` | public | `{"kid","pk","alg"}` — robot ephemeral pubkey (see below) |
| `WIFI_SCAN` | session | JSON array of SSIDs (byte-bounded to one MTU) or `ERROR: …` |
| `WIFI_CONNECT_ENC <json>` | session | `OK: Connecting to <ssid>` or `ERROR: …`; poll `WIFI_STATUS` for outcome |
| `WIFI_FORGET <ssid>` | session | `OK: Forgotten <ssid>` or `ERROR: …` |

`WIFI_STATUS` withholds the saved-network list from unauthenticated peers (it's
an owner-location fingerprint). `WIFI_KEYEX` is public because a bare public
key is useless without the PIN.

## Sealed password scheme — `x25519-hkdf-sha256-aesgcm`

The WiFi password is **never** transmitted in cleartext (App-Store-review
requirement). BLE pairing here is Just-Works (the robot is `NoInputNoOutput`
hardware, so MITM-protected pairing isn't possible) — so we encrypt at the
application layer and mix the device PIN into the key derivation.

> **Threat model in one line:** this scheme protects the PSK against a
> **passive** eavesdropper, but **not** against an **active** man-in-the-middle
> present during the brief setup exchange. See "Security properties" below —
> closing the active-MITM gap requires a PAKE and is tracked as follow-up work.

Wire flow:

1. `WIFI_KEYEX` → robot returns `{kid, pk}` where `pk` = base64 of its
   ephemeral X25519 public key (32 raw bytes). The keypair rotates every
   ~10 min; a `kid` mismatch later returns `ERROR: Bad credentials` → re-`KEYEX`.
2. Phone generates its **own** ephemeral X25519 keypair, does ECDH against the
   robot key, and derives an AES-256-GCM key:
   - `key = HKDF-SHA256(ecdh_shared, salt = PIN_utf8, info = "reachy-mini-wifi-psk-v1", L = 32)`
3. Phone seals the password: `AES-GCM(key, nonce, plaintext = psk, aad = ssid_utf8)`.
4. Phone writes `WIFI_CONNECT_ENC ` + a JSON blob:
   ```json
   {"ssid":"…","kid":"…","epk":"<b64 phone pubkey 32B>",
    "nonce":"<b64 12B>","ct":"<b64 ciphertext||16B tag>"}
   ```
5. Daemon repeats ECDH+HKDF (it computes the **same** PIN locally), decrypts,
   and connects via `nmcli`.

### Security properties

What this scheme **does** give you:

- **Passive sniffer** sees only public keys + ciphertext → can't recover the
  PSK. Without an ECDH private key there is no shared secret to feed HKDF, so
  the PIN salt is irrelevant and the password stays sealed. This is the
  property the cleartext-avoidance requirement actually cares about, and it
  holds.
- **AAD = ssid** binds the sealed PSK to its target network — a captured blob
  can't be replayed to seal-connect a *different* SSID.
- **Tamper / wrong PIN at the robot** → AES-GCM auth fails → daemon returns
  400 → BLE returns `ERROR: Bad credentials (wrong PIN?)`. Online PIN guessing
  against the robot is separately rate-limited (see "Auth / session" and the
  wrong-PIN throttle in `bluetooth_service.py`).

> ### ⚠️ Known limitation — active MITM can recover the PSK (offline PIN brute-force)
>
> **This scheme does NOT defend against an active man-in-the-middle** within
> BLE range during the few seconds of setup. Mixing a low-entropy PIN into an
> otherwise-unauthenticated key exchange is not enough. The attack:
>
> 1. The phone never verifies it's talking to the *real* robot (Just-Works, no
>    device identity key). On `WIFI_KEYEX` the MITM answers with **its own**
>    X25519 pubkey.
> 2. The phone now shares an ECDH secret with the **attacker**, who therefore
>    knows that secret in full.
> 3. The phone seals the PSK with `HKDF(ecdh_secret, salt = PIN)` and sends it;
>    the MITM captures the ciphertext.
> 4. The only unknown left is the **5-char PIN**. The attacker brute-forces it
>    **offline** on their own machine — for each candidate PIN, re-derive the
>    key and try to open the AES-GCM tag. ~10⁵ guesses is a fraction of a
>    second, and it never touches the robot again, so the wrong-PIN throttle
>    (which only gates *online* guesses) does **not** help here.
>
> Net: an attacker present during provisioning can recover the home WiFi
> password — the exact thing this feature is meant to prevent. The earlier
> "active MITM can't derive the key without the PIN" claim was **wrong** and
> has been removed.
>
> **Fix (tracked as follow-up):** replace the ECDH-then-flavor-with-PIN step
> with a **PAKE** (Password-Authenticated Key Exchange, e.g. CPace or SPAKE2)
> that feeds the PIN *into* the key agreement. A party that doesn't know the
> PIN ends up with a useless key and **cannot** test guesses offline — every
> guess becomes a fresh live attempt the robot can see and throttle. The rest
> of the scheme is unaffected: keep AES-GCM sealing the PSK and the SSID as
> AAD, and only swap the key-agreement half. The PIN already exists and is
> printed on the robot, so no new manufacturing or backend infrastructure is
> needed. (An alternative — a permanent factory device identity key the phone
> pins — also closes the gap but does require manufacturing changes.)

### iOS reference client (CryptoKit — no third-party deps)

```swift
import CryptoKit
import Foundation

/// Seal a WiFi password for `WIFI_CONNECT_ENC`. `robotPkB64`/`kid` come from
/// the `WIFI_KEYEX` reply; `pin` is what the user typed for `PIN_…`.
func sealWifiPassword(ssid: String, psk: String, pin: String,
                      robotPkB64: String, kid: String) throws -> String {
    let robotPub = try Curve25519.KeyAgreement.PublicKey(
        rawRepresentation: Data(base64Encoded: robotPkB64)!)

    let myPriv = Curve25519.KeyAgreement.PrivateKey()
    let shared = try myPriv.sharedSecretFromKeyAgreement(with: robotPub)

    // HKDF must mirror the daemon EXACTLY: salt = PIN, info = label, 32 bytes.
    let key = shared.hkdfDerivedSymmetricKey(
        using: SHA256.self,
        salt: Data(pin.utf8),
        sharedInfo: Data("reachy-mini-wifi-psk-v1".utf8),
        outputByteCount: 32)

    // AAD = ssid binds the ciphertext to the target network.
    let sealed = try AES.GCM.seal(Data(psk.utf8), using: key,
                                  authenticating: Data(ssid.utf8))

    let blob: [String: String] = [
        "ssid": ssid,
        "kid": kid,
        "epk": myPriv.publicKey.rawRepresentation.base64EncodedString(),
        "nonce": Data(sealed.nonce).base64EncodedString(),       // 12 bytes
        "ct": (sealed.ciphertext + sealed.tag).base64EncodedString(), // ct||tag
    ]
    let json = String(data: try JSONSerialization.data(withJSONObject: blob), encoding: .utf8)!
    return "WIFI_CONNECT_ENC " + json
}
```

Notes:
- CryptoKit's auto-generated nonce is 12 bytes; send `sealed.nonce`.
- `ct` must be `ciphertext + tag` (Python's `AESGCM.decrypt` expects the 16-byte
  tag appended). CryptoKit exposes them separately — concatenate.
- Salt/info/AAD must match byte-for-byte or decryption fails by design.

## Error strings (Response notifications)

| String | Meaning |
|---|---|
| `ERROR: Not connected. Please authenticate first.` | no live session — send `PIN_…` |
| `ERROR: Bad credentials (wrong PIN?)` | sealed decrypt failed (PIN mismatch / tamper / stale `kid`) |
| `ERROR: Busy` | an nmcli op is already running (daemon 409) |
| `ERROR: Unknown ssid` / `ERROR: Cannot forget hotspot` | `WIFI_FORGET` 404 / 400 |
| `ERROR: Daemon unreachable` | daemon HTTP not answering on localhost |

## Where the code lives

- BLE relay + dispatch: `bluetooth_service.py` (`_handle_command`, `_wifi_*`).
- Crypto + nmcli: `daemon/app/routers/wifi_config.py`
  (`get_provisioning_key`, `connect_to_wifi_network_sealed`, `_open_sealed_psk`).
