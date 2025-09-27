import Link from "next/link"
import { CopyButton } from "@/components/copy-button"
import { DiscordFab } from "@/components/discord-fab"
import { GlobeSection } from "@/components/globe-section"
import { DemoGif } from "@/components/demo-gif"
import { FrameworkCarousel } from "@/components/framework-carousel"

export default function Home() {
  return (
    <>
    <main className="min-h-screen px-6 py-16">
      <section className="mx-auto w-full max-w-6xl text-center hero-surface p-12 md:p-20" id="hero">
        <h1 className="mt-4 text-4xl md:text-7xl font-semibold tracking-tight">Flower + RL = Federated RL</h1>
        <p className="mt-4 text-base md:text-lg text-gray-300">
          A unified toolkit for the RL research world
        </p>
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
          <a href="#quickstart" className="btn-accent rounded-md px-4 py-2 text-sm font-medium">Quickstart</a>
        </div>
        
        {/* Demo GIF placeholder */}
        <div className="mt-8 mx-auto max-w-4xl">
          <div className="rounded-xl border border-white/10 bg-black/40 p-6">
            <div className="flex items-center gap-2 text-sm text-gray-400 mb-4">
            </div>
            <div className="aspect-video bg-black/60 rounded-lg border border-white/10 flex items-center justify-center">
              <DemoGif />
            </div>
          </div>
        </div>
      </section>

      <section id="quickstart" className="mx-auto w-full max-w-4xl mt-12 reveal">
        <h2 className="text-2xl md:text-3xl font-semibold">Quickstart</h2>
        <div className="mt-3 rounded-lg border border-black/10 dark:border-white/10 bg-white/50 dark:bg-black/20 p-4">
          <pre className="whitespace-pre-wrap break-words text-sm md:text-[13px] text-gray-800 dark:text-gray-200">pip install -e . && flwr run .</pre>
          <div className="mt-3"><CopyButton text="pip install -e . && flwr run ." label="Copy" /></div>
        </div>
      </section>

      {/* The Problem: Flower's RL Gap */}
      <section id="problem" className="mx-auto w-full max-w-6xl mt-16 reveal">
        <div className="rounded-xl border border-white/10 bg-black/40 p-8">
          <h2 className="text-2xl md:text-3xl font-semibold mb-6">The Problem: Flower&apos;s RL Gap</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-green-400">Flower today: Great for this</h3>
              <pre className="bg-black/60 p-4 rounded-lg text-sm text-gray-200 overflow-auto">
{`train_epoch(dataloader) # Fixed batches, i.i.d. data
loss, accuracy = evaluate(test_set)`}
              </pre>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-red-400">But struggles with this</h3>
              <pre className="bg-black/60 p-4 rounded-lg text-sm text-gray-200 overflow-auto">
{`while not done:
    action = policy(state) # Sequential, on-policy
    state, reward = env.step(action) # Non-i.i.d. trajectories`}
              </pre>
            </div>
          </div>
        </div>
      </section>

      {/* Our Solution: Complete FRL Stack */}
      <section id="solution" className="mx-auto w-full max-w-6xl mt-16 reveal">
        <div className="rounded-xl border border-white/10 bg-black/40 p-8">
          <h2 className="text-2xl md:text-3xl font-semibold mb-6">Our Solution: Complete FRL Stack</h2>
          
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-blue-400">After our contribution</h3>
            <pre className="bg-black/60 p-4 rounded-lg text-sm text-gray-200 overflow-auto">
{`rl_client = A2CClient(config)
adapter = FlowerClientAdapter(rl_client)
fl.client.start_numpy_client(server, adapter) # Just works!`}
            </pre>
          </div>
        </div>
      </section>

      {/* FRL Challenges */}
      <section id="challenges" className="mx-auto w-full max-w-6xl mt-16 reveal">
        <div className="rounded-xl border border-white/10 bg-black/40 p-8">
          <h2 className="text-2xl md:text-3xl font-semibold mb-6">FRL Challenges We Solve</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">RL Round Semantics</h3>
              <p className="text-gray-400">Training driven by env steps/episodes, not fixed mini-batches. No native &quot;steps per round&quot; contract.</p>
              
              <h3 className="text-lg font-semibold">Trajectory Privacy</h3>
              <p className="text-gray-400">RL data are on-policy and streaming; examples aren&apos;t i.i.d. Current examples assume dataset-style batches.</p>
              
              <h3 className="text-lg font-semibold">Model Packaging</h3>
              <p className="text-gray-400">RL needs PyTorch state_dict blobs (actor+critic, opt state)—not simple NumPy arrays for each layer.</p>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Metrics for Control</h3>
              <p className="text-gray-400">RL needs entropy, KL, value loss, episode returns—beyond standard loss/accuracy.</p>
              
              <h3 className="text-lg font-semibold">Personalization</h3>
              <p className="text-gray-400">Non-IID sites need global backbone + local heads (policy/value) rather than averaging everything.</p>
              
              <h3 className="text-lg font-semibold">Eval Flow</h3>
              <p className="text-gray-400">FRL demos require easy &quot;Evaluate N episodes / Deploy checkpoint&quot; path, which isn&apos;t baked in.</p>
            </div>
          </div>
        </div>
      </section>

      <FrameworkCarousel />

      <GlobeSection />

      <footer className="mx-auto w-full max-w-5xl mt-16 flex items-center justify-between border-t border-black/10 dark:border-white/10 pt-6">
        <p className="text-sm text-gray-600 dark:text-gray-400">© {new Date().getFullYear()} flwr-frl-kit</p>
      </footer>
    </main>
    <DiscordFab />
    </>
  )
}
