/**
 * Loopback transport for code that runs ON the robot.
 *
 * In a browser the SDK talks to the daemon over a WebRTC DataChannel. A JS
 * app's robot module runs on the robot itself (under Deno, started by the
 * daemon) and reaches the same command surface over a loopback WebSocket, the
 * daemon's "app link" (see `reachy_mini/daemon/app_link.py`). Same JSON
 * messages both ways, so every SDK method behaves identically on either side.
 */

/** The slice of `RTCDataChannel` the SDK sends commands through. */
export interface CommandChannel {
    readonly readyState: string;
    readonly bufferedAmount: number;
    send(data: string): void;
    close(): void;
}

export interface LocalLinkHandlers {
    onMessage(text: string): void;
    onClose(reason: string): void;
}

/** Open the daemon's app link and adapt it to a {@link CommandChannel}. */
export function openLocalLink(url: string, handlers: LocalLinkHandlers): Promise<CommandChannel> {
    return new Promise((resolve, reject) => {
        const ws = new WebSocket(url);
        let opened = false;
        const channel: CommandChannel = {
            get readyState() {
                return ws.readyState === WebSocket.OPEN ? 'open' : 'closed';
            },
            get bufferedAmount() {
                return ws.bufferedAmount;
            },
            send: (data) => ws.send(data),
            close: () => ws.close(),
        };
        ws.onopen = () => {
            opened = true;
            resolve(channel);
        };
        ws.onmessage = (ev) => {
            if (typeof ev.data === 'string') handlers.onMessage(ev.data);
        };
        ws.onerror = () => {
            if (!opened) reject(new Error(`cannot open robot link ${url.split('?')[0]}`));
        };
        ws.onclose = (ev) => {
            if (opened) handlers.onClose(ev.reason || `code ${ev.code}`);
        };
    });
}
