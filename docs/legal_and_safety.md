# Legal and safety notes

## TL;DR

**Do not run this bot on any VAC-protected server.** Doing so will result in a
permanent VAC ban on your Steam account, which cannot be appealed.

## What is "VAC-protected"?

VAC (Valve Anti-Cheat) is enabled on:

* Official Valve matchmaking (Premier, Competitive, Wingman, Deathmatch,
  Casual, Arms Race).
* Most community servers (those running with `+sv_pure 1` and VAC enabled).
* Third-party platforms with their own anti-cheat (FACEIT AC, ESEA Client,
  ChallengerMode, etc.) — these are *separate* anti-cheats, but they will
  detect this bot just as easily.

Any input simulation (`SendInput`, `xdotool`, `pynput`) or memory reading
(unrelated to this scaffold) on a VAC-protected server is detectable and
sanctionable. Even if the bot loses every round, the input pattern itself
(superhuman mouse precision, perfectly timed key releases) is a strong signal.

## What is safe?

* **Local server with `sv_cheats 1`:** start a local lobby with bots
  (`bot_add t`, `bot_add ct`) on `de_dust2` or any custom map. No anti-cheat,
  no Steam profile flags.
* **LAN dedicated server with `sv_lan 1`:** run a private CS2 server on a
  machine you own and connect over LAN. Disable VAC explicitly.
* **CSGO Workshop maps in offline mode.**

## Risk acknowledgement

The training entrypoint refuses to start unless the operator passes
`--i-understand-the-risks`. This is a deliberate speed bump, not a legal
shield. Running this software is your responsibility.

## Kill switch

The controller respects a hard kill-switch toggled by F9 (configurable via
`controller.safety_kill_key`). When engaged, all currently-held keys are
released and further actions are dropped until cleared. Use this if the bot
behaves unexpectedly during a session — you should always have a physical
keyboard/mouse within reach.

## Data handling

* The GSI listener binds to `127.0.0.1` by default — payloads never leave your
  machine.
* The auth token in `cfg/default.yaml` is a placeholder. Change it before use
  and treat it as a secret.
* Screen captures are kept in-memory unless you explicitly opt in to logging
  them to disk. Be careful when sharing recordings: they may contain account
  IDs, IP addresses (in console output), and other personally-identifying
  information.
