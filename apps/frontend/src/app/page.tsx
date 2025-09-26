import Link from "next/link"
import { CopyButton } from "@/components/copy-button"
import { DiscordFab } from "@/components/discord-fab"
import { GlobeSection } from "@/components/globe-section"

export default function Home() {
  return (
    <>
    <main className="min-h-screen px-6 py-16">
      <section className="mx-auto w-full max-w-6xl text-center hero-surface p-12 md:p-20">
        <div className="inline-flex items-center gap-2 rounded-full border border-black/10 dark:border-white/10 px-3 py-1 text-xs text-gray-600 dark:text-gray-300">
          <span>Project</span>
        </div>
        <h1 className="mt-4 text-4xl md:text-7xl font-semibold tracking-tight">YOUR TEAM. ONE BRAIN.</h1>
        <p className="mt-4 text-base md:text-lg text-gray-300">
          flwr-frl-kit is a minimal, production-ready Federated RL scaffolding + Strategy for Flower.
        </p>
        <p className="mt-2 text-sm md:text-base text-gray-400">
          Gymnasium-friendly client template, delay-tolerant A2C/A3C aggregation, per-site returns, and secure-agg stubs.
        </p>
        <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <input placeholder="Enter your email" className="w-full max-w-xs rounded-md border border-white/10 bg-black/40 px-3 py-2 text-sm placeholder:text-gray-500" />
          <button className="btn-primary rounded-md px-4 py-2 text-sm font-medium">Get Beta Access</button>
        </div>
      </section>

      <section id="why-track3" className="mx-auto w-full max-w-4xl mt-16 space-y-3">
        <h2 className="text-2xl md:text-3xl font-semibold">Why this fits Track-3</h2>
        <p className="text-gray-700 dark:text-gray-300">
          It’s infrastructure (Strategy + Mod + examples) rather than a model result, so it’s PR-able. It uses Flower’s Strategy/Message API and runs in Simulation or Deployment Engine (SuperLink/SuperNodes).
        </p>
      </section>

      <section id="what-it-solves" className="mx-auto w-full max-w-4xl mt-12 space-y-3">
        <h2 className="text-2xl md:text-3xl font-semibold">What it solves (plain)</h2>
        <p className="text-gray-700 dark:text-gray-300">
          FRL teams keep re-inventing boilerplate to make RL work with FL frameworks (handling streaming, non-IID, staleness). Flower has great building blocks, but no official, batteries-included FRL starter.
        </p>
        <p className="text-gray-700 dark:text-gray-300">
          Policy rollout locally → aggregate policy/value heads server-side with staleness-aware logic.
        </p>
      </section>

      <section id="kit" className="mx-auto w-full max-w-4xl mt-12 space-y-4">
        <h2 className="text-2xl md:text-3xl font-semibold">The kit (deliverables)</h2>
        <ul className="list-disc pl-6 space-y-2 text-gray-700 dark:text-gray-300">
          <li><strong>FRLClientApp (Gymnasium)</strong>: rollouts N episodes, computes local A2C grads/updates, logs returns/lengths/violations, replies with ArrayRecord(s)+MetricRecord via Message API.</li>
          <li><strong>DelayAwareA2CStrategy</strong>: sync or semi-async aggregation; down-weights stale clients by age/lag; tracks global return and client contribution.</li>
          <li><strong>SecureAggStubMod</strong>: API-compatible masking stub to later swap in secure aggregation/DP.</li>
          <li><strong>Examples</strong>: Simulation (Ray) and Deployment Engine (SuperLink/SuperNodes).</li>
          <li><strong>Dashboards</strong>: W&B (or CSV) for returns vs rounds, staleness vs weight, time/round, failures.</li>
          <li><strong>1-click scripts</strong>: sim and deploy flows.</li>
        </ul>
      </section>

      <section id="plan" className="mx-auto w-full max-w-4xl mt-12 space-y-4">
        <h2 className="text-2xl md:text-3xl font-semibold">6-hour build plan</h2>
        <ol className="list-decimal pl-6 space-y-2 text-gray-700 dark:text-gray-300">
          <li><strong>Hour 0–1 — Scaffold & baselines</strong>: skeleton via Flower, add Gymnasium CartPole + tiny A2C loop.</li>
          <li><strong>Hour 1–2 — Strategy (sync)</strong>: implement DelayAwareA2CStrategy with lag-based weights (e.g., w_i ∝ exp(-alpha * staleness_i)).</li>
          <li><strong>Hour 2–3 — Simulation</strong>: run with Simulation Engine (Ray), validate rounds, returns, timing.</li>
          <li><strong>Hour 3–4 — Deployment</strong>: start SuperLink + two SuperNodes, confirm identical training flow.</li>
          <li><strong>Hour 4–5 — Secure-agg + logging</strong>: masking stub, CSV/W&B metrics.</li>
          <li><strong>Hour 5–6 — Polish & demo</strong>: README, dashboards, demo scripts.</li>
        </ol>
      </section>

      <section id="demo" className="mx-auto w-full max-w-4xl mt-12 space-y-3">
        <h2 className="text-2xl md:text-3xl font-semibold">1-minute demo script</h2>
        <p className="text-gray-700 dark:text-gray-300">
          Hook (10s): FRL is hard on real networks... We built flwr-frl-kit.
        </p>
        <p className="text-gray-700 dark:text-gray-300">
          How (20s): Clients roll out episodes, compute A2C updates; strategy aggregates with staleness-aware weights.
        </p>
        <p className="text-gray-700 dark:text-gray-300">
          Proof (20s): Simulation finishes 3 rounds; Deployment over SuperLink/SuperNodes — no code change.
        </p>
        <p className="text-gray-700 dark:text-gray-300">Impact (10s): Prototype FRL quickly with hooks for secure agg/DP.</p>
      </section>

      <section id="commands" className="mx-auto w-full max-w-4xl mt-12">
        <h2 className="text-2xl md:text-3xl font-semibold">Quickstart</h2>
        <div className="mt-3 rounded-lg border border-black/10 dark:border-white/10 bg-white/50 dark:bg-black/20 p-4">
          <pre className="whitespace-pre-wrap break-words text-sm md:text-[13px] text-gray-800 dark:text-gray-200">pip install -e . && flwr run .</pre>
          <div className="mt-3"><CopyButton text="pip install -e . && flwr run ." label="Copy" /></div>
        </div>
      </section>

      {/* Feature two-column panel */}
      <section id="features" className="mx-auto w-full max-w-6xl mt-16">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 rounded-xl border border-white/10 bg-black/40 p-6 md:p-8">
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-500" />
              <span>Institutional Memory</span>
            </div>
            <h3 className="text-xl md:text-2xl font-semibold">Your team's knowledge, always accessible.</h3>
            <p className="text-gray-400">Context flows seamlessly from simulation to deployment, preserving the "why" behind every change.</p>
            <div className="rounded-lg border border-white/10 bg-black/30 p-4 text-sm">
              <div className="text-gray-400">user-prompt/ui-update</div>
              <div className="mt-2 font-medium">Add responsive navigation bar to my app</div>
            </div>
          </div>
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-500" />
              <span>MCP Integration</span>
            </div>
            <h3 className="text-xl md:text-2xl font-semibold">Context everywhere you code. Works in any IDE.</h3>
            <p className="text-gray-400">Built on Flower Strategy/Message API. Use the tools you are comfortable with.</p>
            <div className="relative rounded-lg border border-white/10 bg-black/30 p-8">
              <div className="absolute inset-0 pointer-events-none" style={{background:"radial-gradient(60% 60% at 50% 60%, rgba(239,68,68,0.08), transparent 60%)"}} />
              <div className="relative flex items-center gap-4 opacity-90">
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/10">VS</span>
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/10">Py</span>
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-white/10">W&B</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Collaboration section */}
      <section id="collaboration" className="mx-auto w-full max-w-6xl mt-16">
        <div className="rounded-xl border border-white/10 bg-black/40 p-8">
          <h3 className="text-2xl md:text-3xl font-semibold">Shared context, workflows & threads.</h3>
          <p className="mt-2 text-gray-400">Reuse what works, track adoption, and improve together across simulation and deployment.</p>
        </div>
      </section>

      {/* Partners row */}
      <section id="partners" className="mx-auto w-full max-w-6xl mt-12">
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

      <footer className="mx-auto w-full max-w-5xl mt-16 flex items-center justify-between border-t border-black/10 dark:border-white/10 pt-6">
        <p className="text-sm text-gray-600 dark:text-gray-400">© {new Date().getFullYear()} flwr-frl-kit</p>
        <div className="flex items-center gap-4 text-sm">
          <Link href="#kit" className="hover:underline">Kit</Link>
          <Link href="#plan" className="hover:underline">Plan</Link>
          <Link href="#demo" className="hover:underline">Demo</Link>
        </div>
      </footer>
    </main>
    <DiscordFab />
    </>
  )
}
