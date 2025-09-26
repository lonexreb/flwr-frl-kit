import Link from "next/link"
import { CopyButton } from "@/components/copy-button"
import { DiscordFab } from "@/components/discord-fab"
import { GlobeSection } from "@/components/globe-section"

export default function Home() {
  return (
    <>
    <main className="min-h-screen px-6 py-16">
      <section className="mx-auto w-full max-w-6xl text-center hero-surface p-12 md:p-20" id="hero">
        <div className="inline-flex items-center gap-2 rounded-full border border-black/10 dark:border-white/10 px-3 py-1 text-xs text-gray-400">
          <span>Federated Reinforcement Learning</span>
        </div>
        <h1 className="mt-4 text-4xl md:text-7xl font-semibold tracking-tight">Sleek FRL. Ship faster.</h1>
        <p className="mt-4 text-base md:text-lg text-gray-300">
          Minimal kit for Flower: rollouts on clients, staleness-aware aggregation on server, deploy anywhere.
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          <a href="#quickstart" className="btn-accent rounded-md px-4 py-2 text-sm font-medium">Quickstart</a>
          <a href="#globe" className="btn-primary rounded-md px-4 py-2 text-sm font-medium">See Network</a>
        </div>
      </section>

      <section id="why" className="mx-auto w-full max-w-6xl mt-16 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="reveal rounded-xl border border-white/10 bg-black/40 p-6">
          <h3 className="text-lg font-semibold">Production-first</h3>
          <p className="mt-1 text-gray-400">Strategy + client runtime, not a toy repo. Works in sim and deploy.</p>
        </div>
        <div className="reveal rounded-xl border border-white/10 bg-black/40 p-6">
          <h3 className="text-lg font-semibold">Flower-native</h3>
          <p className="mt-1 text-gray-400">Uses Strategy/Message API. Keep your tools and infra.</p>
        </div>
        <div className="reveal rounded-xl border border-white/10 bg-black/40 p-6">
          <h3 className="text-lg font-semibold">Delay-tolerant</h3>
          <p className="mt-1 text-gray-400">Staleness-aware weights. Non-IID friendly.</p>
        </div>
      </section>

      <section id="blocks" className="mx-auto w-full max-w-6xl mt-12 grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="reveal rounded-xl border border-white/10 bg-black/40 p-6">
          <h3 className="text-lg font-semibold">Client</h3>
          <p className="mt-1 text-gray-400">Gymnasium rollouts → A2C updates.</p>
        </div>
        <div className="reveal rounded-xl border border-white/10 bg-black/40 p-6">
          <h3 className="text-lg font-semibold">Server</h3>
          <p className="mt-1 text-gray-400">Aggregate with lag-aware weights.</p>
        </div>
        <div className="reveal rounded-xl border border-white/10 bg-black/40 p-6">
          <h3 className="text-lg font-semibold">Secure agg</h3>
          <p className="mt-1 text-gray-400">Masking stub, DP-ready.</p>
        </div>
      </section>

      <section id="kit" className="mx-auto w-full max-w-6xl mt-12 grid grid-cols-1 md:grid-cols-3 gap-4 reveal">
        <div className="rounded-xl border border-white/10 bg-black/40 p-6">
          <h3 className="text-lg font-semibold">Client runtime</h3>
          <p className="mt-1 text-gray-400">Gymnasium rollouts, A2C updates, clean logs.</p>
        </div>
        <div className="rounded-xl border border-white/10 bg-black/40 p-6">
          <h3 className="text-lg font-semibold">Strategy</h3>
          <p className="mt-1 text-gray-400">Staleness-aware aggregation, sync or semi-async.</p>
        </div>
        <div className="rounded-xl border border-white/10 bg-black/40 p-6">
          <h3 className="text-lg font-semibold">Secure agg (stub)</h3>
          <p className="mt-1 text-gray-400">Masking interface, DP-ready hooks.</p>
        </div>
        <div className="rounded-xl border border-white/10 bg-black/40 p-6">
          <h3 className="text-lg font-semibold">Examples</h3>
          <p className="mt-1 text-gray-400">Simulation (Ray) and deployment recipes.</p>
        </div>
        <div className="rounded-xl border border-white/10 bg-black/40 p-6">
          <h3 className="text-lg font-semibold">Dashboards</h3>
          <p className="mt-1 text-gray-400">Returns, staleness, timing, failures.</p>
        </div>
        <div className="rounded-xl border border-white/10 bg-black/40 p-6">
          <h3 className="text-lg font-semibold">1‑click scripts</h3>
          <p className="mt-1 text-gray-400">Run sim and deploy flows quickly.</p>
        </div>
      </section>

      {/* Terminal ASCII art */}
      <section id="terminal" className="mx-auto w-full max-w-6xl mt-12 reveal">
        <div className="rounded-xl border border-white/10 bg-black/40 p-6">
          <pre className="text-[12px] leading-5 text-gray-200 whitespace-pre-wrap">
{`     ______ _                _____ _____ _     _   _ _ _   
    |  ___| |              |  ___|  _  | |   | | | (_) |  
    | |_  | | ___  ___ _ __| |__ | | | | |   | | | |_| |_ 
    |  _| | |/ _ \/ _ \ '__|  __|| | | | |   | | | | | __|
    | |   | |  __/  __/ |  | |___\ \_/ / |___\ \_/ / | |_ 
    \_|   |_|\___|\___|_|  \____/ \___/\____/ \___/|_|\__|
`}
          </pre>
        </div>
      </section>

      <section id="quickstart" className="mx-auto w-full max-w-4xl mt-12 reveal">
        <h2 className="text-2xl md:text-3xl font-semibold">Quickstart</h2>
        <div className="mt-3 rounded-lg border border-black/10 dark:border-white/10 bg-white/50 dark:bg-black/20 p-4">
          <pre className="whitespace-pre-wrap break-words text-sm md:text-[13px] text-gray-800 dark:text-gray-200">pip install -e . && flwr run .</pre>
          <div className="mt-3"><CopyButton text="pip install -e . && flwr run ." label="Copy" /></div>
        </div>
      </section>

      {/* Feature two-column panel */}
      <section id="features" className="mx-auto w-full max-w-6xl mt-16 reveal">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 rounded-xl border border-white/10 bg-black/40 p-6 md:p-8">
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-500" />
              <span>Institutional Memory</span>
            </div>
            <h3 className="text-xl md:text-2xl font-semibold">Your team's knowledge, always accessible.</h3>
            <p className="text-gray-400">Context flows seamlessly from simulation to deployment, preserving the "why" behind every change.</p>
            <pre className="rounded-lg border border-white/10 bg-black/30 p-4 text-[13px] text-gray-200 overflow-auto">
{`# Strategy weight (server)
def weight(staleness, alpha=0.3):
    return math.exp(-alpha * staleness)

# Client loop (pseudo)
for episode in range(N):
    traj = rollout(env, policy)
    send_update(grad_from(traj))
`}
            </pre>
          </div>
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-500" />
              <span>MCP Integration</span>
            </div>
            <h3 className="text-xl md:text-2xl font-semibold">Context everywhere you code. Works in any IDE.</h3>
            <p className="text-gray-400">Built on Flower Strategy/Message API. Use the tools you are comfortable with.</p>
            <pre className="relative rounded-lg border border-white/10 bg-black/30 p-4 text-[13px] text-gray-200 overflow-auto">
{`# Same API, different algo
client = RLClient(env, algo="sac")  # or "a2c"
client.run(rounds=3)

# Metrics schema (printed by Train)
{
  "round": int,
  "return_mean": float,
  "staleness": float
}
`}
            </pre>
          </div>
        </div>
      </section>

      {/* Collaboration section */}
      <section id="collaboration" className="mx-auto w-full max-w-6xl mt-16 reveal">
        <div className="rounded-xl border border-white/10 bg-black/40 p-8">
          <h3 className="text-2xl md:text-3xl font-semibold">Shared context, workflows & threads.</h3>
          <p className="mt-2 text-gray-400">Reuse what works, track adoption, and improve together across simulation and deployment.</p>
        </div>
      </section>

      {/* Partners row */}
      <section id="partners" className="mx-auto w-full max-w-6xl mt-12 reveal">
        <div className="rounded-xl border border-white/10 bg-black/40 p-4">
          <div className="text-sm text-gray-400">Backed by</div>
          <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 gap-4 opacity-80">
            <div className="h-10 rounded-md border border-white/10 bg-black/30" />
            <div className="h-10 rounded-md border border-white/10 bg-black/30" />
            <div className="h-10 rounded-md border border-white/10 bg-black/30" />
            <div className="h-10 rounded-md border border-white/10 bg-black/30" />
            <div className="h-10 rounded-md border border-white/10 bg-black/30" />
            <div className="h-10 rounded-md border border-white/10 bg-black/30" />
          </div>
        </div>
      </section>

      <GlobeSection />

      {/* General benefits */}
      <section id="benefits" className="mx-auto w-full max-w-6xl mt-16 grid grid-cols-1 md:grid-cols-2 gap-6 reveal">
        <div className="rounded-xl border border-white/10 bg-black/40 p-6 space-y-3">
          <h3 className="text-xl md:text-2xl font-semibold">What you get</h3>
          <ul className="list-disc pl-6 text-gray-300 space-y-2">
            <li>Train prints the exact schema you need</li>
            <li>Checkpoints saved for Deploy → Eval with the harness</li>
            <li>Optional Hydra, unit tests, personalization splitter</li>
            <li>Flip <code>USE_MPS=1</code> for Apple GPU</li>
            <li>Drop-in SAC next, same <code>RLClient</code> API</li>
          </ul>
        </div>
        <div className="rounded-xl border border-white/10 bg-black/40 p-6">
          <h4 className="text-sm text-gray-400">Why it’s better</h4>
          <pre className="mt-2 whitespace-pre-wrap break-words text-[13px] text-gray-200">
{`# Delay-aware weight (server)
w_i = exp(-alpha * staleness_i)

# Client update
adv = R_t - V(s_t)
g = grad(log pi(a_t|s_t) * adv + beta * H(pi))

# Same API for A2C and SAC
client = RLClient(env, algo="a2c")
client.run(rounds=3)
`}
          </pre>
        </div>
      </section>

      <footer className="mx-auto w-full max-w-5xl mt-16 flex items-center justify-between border-t border-black/10 dark:border-white/10 pt-6">
        <p className="text-sm text-gray-600 dark:text-gray-400">© {new Date().getFullYear()} flwr-frl-kit</p>
        <div className="flex items-center gap-4 text-sm">
          <Link href="#kit" className="hover:underline">Kit</Link>
          <Link href="#benefits" className="hover:underline">Benefits</Link>
          <Link href="#globe" className="hover:underline">Network</Link>
        </div>
      </footer>
    </main>
    <DiscordFab />
    </>
  )
}
