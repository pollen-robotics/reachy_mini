export const meta = {
  name: 'reachy-guide-eval',
  description: 'Build the 10 reference user requests from the repo docs alone, then score the docs',
  whenToUse: 'Before/after a change to AGENTS.md, skills/ or the JS/Python app guides, to measure whether the docs still steer fresh agents well.',
  phases: [
    { title: 'Setup', detail: 'snapshot the docs ref into a detached worktree, create one empty dir per app' },
    { title: 'Build', detail: 'one fresh agent per user request, docs snapshot as the only source' },
    { title: 'Synthesize', detail: 'app-type tally vs expectations, friction counts, comparison with a baseline' },
  ],
}

// args: {
//   ref:          git ref whose docs are under test (e.g. "origin/main", "origin/my-docs-branch")
//   repo:         path to a reachy_mini clone to snapshot from
//   workDir:      empty scratch dir; gets <workDir>/reachy_mini (snapshot) and <workDir>/apps/<id>
//   only?:        optional list of prompt ids to run (e.g. ["04", "09"])
//   baselinePath?: optional path to a previous results markdown file to compare against
// }
const A = args || {}
if (!A.ref || !A.repo || !A.workDir) throw new Error('args.ref, args.repo and args.workDir are required')

// Expectations come from the guide's own criteria (AGENTS.md "Two App Flavours").
// Prompts are written as a non-technical user would; they never name a flavour or the original app.
const PROMPTS = [
  { id: '01', slug: 'focus-buddy', expected: 'js', note: 'camera + detection in the browser',
    text: "I keep getting distracted by my phone when I work. I'd like my Reachy Mini to keep an eye on me while I'm at my desk: when it sees me picking up my phone it should react (a disapproving look, a little comment), and cheer me on when I stay focused for a while. I'd like to tweak how strict it is and see what it's doing from my laptop or my phone." },
  { id: '02', slug: 'race-commentator', expected: 'js', note: 'public data API + TTS + head moves',
    text: "I love Formula 1. I'd like Reachy Mini to be a race commentator: I pick a past race (or a short demo), and it replays what happened in the race and commentates it out loud with an excited voice, moving its head with the action. I want to be able to speed the replay up." },
  { id: '03', slug: 'voice-chat', expected: 'js', note: 'minimal-conversation is exactly this',
    text: "I just want to talk with my Reachy Mini: a natural voice conversation where it answers out loud, reacts with expressions and little dances, looks at me, and can tell me what it sees through its camera when I ask. It would be fun to switch between a few personalities." },
  { id: '04', slug: 'nursery-watch', expected: 'python', note: 'unattended all night, data stays home',
    text: "I have a newborn. I'd like Reachy Mini to sit in the nursery and watch over the baby: if the baby cries, it should try to soothe them (gentle movements, a soft voice or a lullaby) and alert me on my phone with a picture. It has to work on its own all night, and I'd rather nothing about my baby leaves my home." },
  { id: '05', slug: 'control-center', expected: 'mixed', note: 'most features JS; CPU/Wi-Fi health needs robot-side code',
    text: "I just unboxed my Reachy Mini and I want one place that does everything: talk to it, see its camera, play emotions and dances, move it manually, and check on its health (temperature, CPU, network…). Something I can open from any device at home." },
  { id: '06', slug: 'agent-body', expected: 'mixed', note: 'https page calling a localhost API',
    extra: "The user's assistant API is not reachable from here either; assume a simple HTTP chat endpoint on the user's computer and make its URL configurable.",
    text: "I run a personal AI assistant on my computer (it has its own tools and memory, and exposes a local API). I'd like Reachy Mini to become its body: I talk to the robot, my assistant answers through it out loud, and the robot looks at me and moves expressively while it talks." },
  { id: '07', slug: 'live-3d-twin', expected: 'js', note: 'visualization of the pose stream',
    text: "I'd like to see my Reachy Mini live in 3D in a web page: a 3D model that moves exactly like the real robot in real time, that I can rotate and zoom around." },
  { id: '08', slug: 'language-partner', expected: 'js', note: 'voice conversation + persistence',
    text: "I'm learning Spanish. I'd like Reachy Mini to be my conversation partner: we chat out loud in Spanish at my level, it corrects me gently, explains the grammar when I ask why, and remembers my progress and my recurring mistakes between sessions. It should celebrate when I do well." },
  { id: '09', slug: 'smart-home-voice', expected: 'python', note: 'always-on LAN service Home Assistant must reach',
    extra: 'No Home Assistant instance is reachable from here; make its address configurable.',
    text: "I use Home Assistant for my smart home. I'd like Reachy Mini to become a voice assistant for it: I say a wake word, ask it to turn on the lights and so on, and it answers out loud with expressive movements. It should show up in Home Assistant so I can use it in my automations." },
  { id: '10', slug: 'teach-by-hand', expected: 'js', note: 'an official JS port (marionette-js) exists',
    text: "I'd like to teach my Reachy Mini new moves by hand: I physically move its head while it's relaxed, it records the motion (and the sound at the same time), and then it can replay any recording. I want to manage my recordings and maybe share them." },
]
const selected = A.only ? PROMPTS.filter((p) => A.only.includes(p.id)) : PROMPTS
if (A.only) log(`Running ${selected.length}/${PROMPTS.length} prompts: ${selected.map((p) => p.id).join(', ')}`)

const snapshot = `${A.workDir}/reachy_mini`
const appDir = (p) => `${A.workDir}/apps/${p.id}-${p.slug}`

phase('Setup')
await agent(
  `Prepare an evaluation workspace. Run exactly these steps and report what you did:\n` +
  `1. git -C ${A.repo} fetch --quiet origin\n` +
  `2. git -C ${A.repo} worktree add --detach ${snapshot} ${A.ref}  (fail loudly if ${snapshot} already exists)\n` +
  `3. mkdir -p ${selected.map(appDir).join(' ')}\n` +
  `Do not modify anything else.`,
  { label: 'setup', phase: 'Setup', effort: 'low' },
)

const REPORT = {
  type: 'object',
  properties: {
    app_type: { type: 'string', enum: ['js', 'python', 'hybrid', 'other'] },
    decided_at: { type: 'string', description: 'when in the process the app type was decided' },
    reasons: { type: 'array', items: { type: 'object', properties: {
      claim: { type: 'string' },
      source: { type: 'string', enum: ['DOC', 'OWN_KNOWLEDGE', 'INFERENCE'] },
      ref: { type: 'string', description: 'path:line "short quote" for DOC sources' },
    }, required: ['claim', 'source'] } },
    rejected_alternatives: { type: 'string', description: 'other app types considered, and whether the docs state the limitation or it was assumed' },
    frictions: { type: 'array', items: { type: 'object', properties: {
      ref: { type: 'string' }, issue: { type: 'string' }, cost: { type: 'string' },
    }, required: ['issue'] } },
    not_found: { type: 'array', items: { type: 'string' }, description: 'things searched for in the docs and not found' },
    verified: { type: 'string' },
    confidence: { type: 'string', enum: ['low', 'medium-low', 'medium', 'medium-high', 'high'] },
    doc_changes: { type: 'array', items: { type: 'string' }, description: 'the 3 doc changes that would have helped most' },
    report_markdown: { type: 'string', description: 'the full report, sections A-F' },
  },
  required: ['app_type', 'decided_at', 'reasons', 'frictions', 'confidence', 'doc_changes', 'report_markdown'],
}

const builderPrompt = (p, i) => `You are taking part in an evaluation of the Reachy Mini developer documentation. **The documentation is what is being evaluated — not you.** We want to learn whether the docs, on their own, lead an AI coding assistant from a plain user request to a good, working app. There is no answer we are hoping for. Do not try to guess what the experimenters want; behave exactly as a careful assistant would with a real user. When the docs were unclear, misleading, or silent and that cost you time or pushed you toward a choice, that is the most valuable data you can give us — report it honestly, including your own mistakes.

## The user's request (verbatim)

> ${p.text}

## Setup

- The user cloned the Reachy Mini repo here: \`${snapshot}\`. **This snapshot is your only documentation source.** Start where an assistant naturally would (its root \`CLAUDE.md\` / \`AGENTS.md\`) and follow what it tells you.
- Ignore any other copy of reachy_mini on this machine and any CLAUDE.md / AGENTS.md / memory content your context may contain from other paths — it is not part of the test.
- Your working directory for everything you create: \`${appDir(p)}\`. Do not read or write anything else under \`${A.workDir}\` (other directories belong to other participants), and do not install tools outside your working directory.
- Tooling available: node / npm, python3, uv, and the \`reachy-mini-app-assistant\` CLI.${p.extra ? `\n- ${p.extra}` : ''}

## Rules

1. **Network**: you may fetch URLs that the snapshot's docs link to (reference apps, SDK packages, CDN, docs pages) and install packages from npm / PyPI inside your working directory. Do NOT use web search, and do NOT look for existing Reachy Mini apps on Hugging Face beyond what the docs link to. General technical knowledge you already have is fine, but label it as such.
2. **No robot is available.** Do not try to reach \`reachy-mini.local\` or any robot IP, and do not start a daemon or simulator. Verify with builds, type checks, unit tests, serving files locally. Local servers: only ports ${6200 + i * 10}–${6209 + i * 10}. Do not advertise anything on the network (no mDNS/zeroconf registration). Do not use a shared browser pane; if you need a browser, run your own headless one.
3. **Never publish anything**: no git push, no Hugging Face repo/Space creation, no login, no \`--publish\` flag. If the docs instruct a publish/deploy step, skip it and note it.
4. **The user is not reachable.** Wherever the docs tell you to ask the user or wait for approval: write the question in \`plan.md\` with the answer you believe this user would most likely give, mark it \`[assumed]\`, and continue. Never stop to wait.
5. **Scope**: a functional MVP of the core experience. Stop once the core loop is implemented and verified as far as possible without a robot.

## Deliverables

The app (structured the way the docs prescribe) and \`plan.md\` in your working directory. Do not write a REPORT.md; return the report through the structured output, where \`report_markdown\` holds the full report with sections: A. what kind of app you built and when you decided; B. every reason, tagged [DOC path:line "quote"] / [OWN KNOWLEDGE] / [INFERENCE], including rejected alternatives and whether the docs state the limitation; C. reading log and things not found; D. friction with path:line and cost; E. what you verified and your confidence it works on a real robot; F. the 3 doc changes that would have helped most (minimal edits preferred).`

phase('Build')
const built = await pipeline(selected, (p, _p, i) =>
  agent(builderPrompt(p, i), { label: `build ${p.id} ${p.slug}`, phase: 'Build', schema: REPORT })
    .then((r) => (r ? { id: p.id, slug: p.slug, expected: p.expected, expected_note: p.note, ...r } : null)))
const results = built.filter(Boolean)
if (results.length < selected.length) log(`${selected.length - results.length} builder(s) returned nothing; synthesis covers ${results.length}`)

phase('Synthesize')
const synthesis = await agent(
  `You are scoring the Reachy Mini docs (ref ${A.ref}) from ${results.length} independent build reports. ` +
  `Each report comes from a fresh agent that built one user request using only the docs.\n\n` +
  `REPORTS (JSON):\n${JSON.stringify(results.map(({ report_markdown, ...rest }) => rest))}\n\n` +
  (A.baselinePath ? `Read the previous results at ${A.baselinePath} and compare against them.\n\n` : '') +
  `Write a markdown results file with:\n` +
  `1. App-type table: id, request, built, expected (from the "expected" field), and whether the choice came from a DOC reason or from inference.\n` +
  `2. Friction table: cluster the frictions across reports into themes and count how many reports hit each (cite one path:line per theme).` +
  (A.baselinePath ? ` Add a column with the baseline count, and mark which themes the change targeted vs which are controls.` : '') + `\n` +
  `3. Claims of the form "X is not possible / not supported" that came from OWN_KNOWLEDGE or INFERENCE rather than DOC: list them; these are the dangerous ones.\n` +
  `4. New findings, each with the report ids that support it.\n` +
  `5. The smallest doc or SDK changes, ranked by (reports affected × severity); prefer deletions and code fixes over new prose.\n` +
  `Return only the markdown.`,
  { label: 'synthesize', phase: 'Synthesize' },
)

return { ref: A.ref, workDir: A.workDir, results, synthesis }
