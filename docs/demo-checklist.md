# Demo Checklist

A pre-demo checklist to ensure smooth execution of the live demonstration.

## Environment
- [ ] Ollama running
- [ ] `qwen3:8b` available
- [ ] `nomic-embed-text` available
- [ ] TigerGraph reachable
- [ ] `.env` present locally
- [ ] TigerGraph secret valid
- [ ] dashboard starts
- [ ] browser can reach `localhost:8080`

## Before recording
- [ ] close unnecessary applications
- [ ] ensure laptop is plugged in
- [ ] ensure GPU/local inference is stable
- [ ] ensure no stale dashboard process is running
- [ ] confirm dashboard is using current source
- [ ] prepare the demo question
- [ ] have the reference answer 41 available
- [ ] have benchmark dashboard ready

## During demo
- [ ] enter question
- [ ] select Ollama
- [ ] select qwen3:8b
- [ ] run all three
- [ ] show answer
- [ ] show latency
- [ ] show tokens
- [ ] show Agentic trace
- [ ] show stopping reason
- [ ] show benchmark summary

## If live execution is slow
- Do NOT panic or claim failure.
- The live smoke test can be explained as local Qwen3 inference on laptop hardware.
- Do not substitute fabricated numbers.
