import Link from "next/link"
import { CopyButton } from "@/components/copy-button"
import { DemoGif } from "@/components/demo-gif"
import { FrameworkCarousel } from "@/components/framework-carousel"

export default function Home() {
  return (
    <>
    <main className="min-h-screen px-6 py-16">
      <section className="mx-auto w-full max-w-6xl text-center hero-surface p-12 md:p-20" id="hero">
        <h1 className="-mt-6 text-4xl md:text-7xl font-semibold tracking-tight group cursor-default relative">
          <span className="group-hover:opacity-0 transition-opacity duration-300">🌸 + 🏋️ = 🌲</span>
          <span className="absolute top-4 left-0 w-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300"><span className="font-bold text-pink-400 border border-pink-400 px-1 rounded">Flower</span> + RL = Federated RL</span>
        </h1>
        <p className="mt-4 text-base md:text-lg text-gray-300">
          A unified toolkit for RL on the edge with <span className="font-bold text-pink-400 border border-pink-400 px-1 rounded">Flower</span>
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

      {/* <section id="quickstart" className="mx-auto w-full max-w-4xl mt-12 reveal">
        <h2 className="text-2xl md:text-3xl font-semibold">Quickstart</h2>
        <div className="mt-3 rounded-lg border border-black/10 dark:border-white/10 bg-white/50 dark:bg-black/20 p-4">
          <pre className="whitespace-pre-wrap break-words text-sm md:text-[13px] text-gray-800 dark:text-gray-200">pip install -e . && flwr run .</pre>
          <div className="mt-3"><CopyButton text="pip install -e . && flwr run ." label="Copy" /></div>
        </div>
      </section> */}

      <section id="problem" className="mx-auto w-full max-w-6xl mt-16 reveal">
        <div className="rounded-xl border border-white/10 bg-black/40 p-8">
          <div className="flex items-center gap-2 mb-6">
            <span className="text-2xl">🎯</span>
            <h2 className="text-2xl md:text-3xl font-semibold text-white">RL has never been easier.</h2>
          </div>

          <h3 className="text-xl font-semibold text-white mb-4">The Problem: <span className="font-bold text-pink-400 border border-pink-400 px-1 rounded">Flower</span>&apos;s RL Gap</h3>

          <div className="space-y-4">
            <div className="bg-gray-900 rounded-lg border border-gray-600 overflow-hidden">
              <div className="bg-gray-800 border-b border-gray-600 px-4 py-2 flex items-center gap-2 text-sm text-gray-300">
                <span className="bg-blue-600 text-white px-2 py-1 rounded text-xs font-mono">In [1]:</span>
              </div>
              <div className="p-4 bg-gray-900">
                <pre className="text-sm font-mono text-white leading-relaxed">
<span className="text-green-400"># Flower today: Great for this </span>
<br/>
<span className="text-cyan-400">train_epoch</span>(<span className="text-yellow-300">dataloader</span>) <span className="text-gray-400"># Fixed batches, i.i.d. data</span>
<br/>
<span className="text-white">loss</span>, <span className="text-white">accuracy</span> <span className="text-pink-400">=</span> <span className="text-cyan-400">evaluate</span>(<span className="text-yellow-300">test_set</span>)
<br/>
<br/>
<span className="text-green-400"># But struggles with this</span> <br/>
<span className="text-purple-400">while</span> <span className="text-purple-400">not</span> <span className="text-white">done</span>: <br/>
&nbsp;&nbsp;&nbsp;&nbsp;<span className="text-white">action</span> <span className="text-pink-400">=</span> <span className="text-cyan-400">policy</span>(<span className="text-white">state</span>) <span className="text-gray-400"># Sequential, on-policy</span><br/>
&nbsp;&nbsp;&nbsp;&nbsp;<span className="text-white">state</span>, <span className="text-white">reward</span> <span className="text-pink-400">=</span> <span className="text-white">env</span>.<span className="text-cyan-400">step</span>(<span className="text-white">action</span>) <span className="text-gray-400"># Non-i.i.d. trajectories</span>
                </pre>
              </div>
            </div>
          </div>

          <h3 className="text-xl font-semibold text-white mt-8 mb-4">Our Solution: Complete FRL Stack</h3>

          <div className="space-y-4">
            <div className="bg-gray-900 rounded-lg border border-gray-600 overflow-hidden">
              <div className="bg-gray-800 border-b border-gray-600 px-4 py-2 flex items-center gap-2 text-sm text-gray-300">
                <span className="bg-blue-600 text-white px-2 py-1 rounded text-xs font-mono">In [2]:</span>
              </div>
              <div className="p-4 bg-gray-900">
                <pre className="text-sm font-mono text-white leading-relaxed">
<span className="text-green-400"># Our Solution: Complete FRL Stack</span>
<br/>
<span className="text-green-400"># After our contribution</span>
<br/>
<span className="text-white">rl_client</span> <span className="text-pink-400">=</span> <span className="text-cyan-400">A2CClient</span>(<span className="text-yellow-300">config</span>)<br/>
<span className="text-white">adapter</span> <span className="text-pink-400">=</span> <span className="text-cyan-400">FlowerClientAdapter</span>(<span className="text-white">rl_client</span>)<br/>
<span className="text-white">fl</span>.<span className="text-white">client</span>.<span className="text-cyan-400">start_numpy_client</span>(<span className="text-white">server</span>, <span className="text-white">adapter</span>) <span className="text-gray-400"># Just works!</span>
                </pre>
              </div>
            </div>
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


      <footer className="mx-auto w-full max-w-5xl mt-16 flex items-center justify-between border-t border-black/10 dark:border-white/10 pt-6">
        <p className="text-sm text-gray-600 dark:text-gray-400">© {new Date().getFullYear()} flwr-frl-kit</p>
      </footer>
    </main>
    </>
  )
}
