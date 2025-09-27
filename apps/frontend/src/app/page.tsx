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
