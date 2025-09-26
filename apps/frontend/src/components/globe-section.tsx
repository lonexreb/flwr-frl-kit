"use client";

import { useEffect, useRef } from "react"
import dynamic from "next/dynamic"

const ThreeGlobe = dynamic(() => import("three-globe"), { ssr: false })

export interface GlobeSectionProps {}

export function GlobeSection(_: GlobeSectionProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const globeRef = useRef<any>(null)

  useEffect(() => {
    let renderer: any
    let scene: any
    let camera: any
    let animationId: number | undefined

    async function setup() {
      const THREE = await import("three")
      const Globe = (await import("three-globe")).default

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
      renderer.setSize(containerRef.current!.clientWidth, 420)
      renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1))

      scene = new THREE.Scene()
      camera = new THREE.PerspectiveCamera(55, (containerRef.current!.clientWidth) / 420, 0.1, 1000)
      camera.position.z = 300

      const globe = new Globe()
        .globeImageUrl("//unpkg.com/three-globe/example/img/earth-dark.jpg")
        .bumpImageUrl("//unpkg.com/three-globe/example/img/earth-topology.png")
        .arcsData(sampleArcs)
        .arcColor(() => ["#ef4444", "#ef4444"]) // red
        .arcAltitude(() => Math.random() * 0.3 + 0.1)
        .arcStroke(() => 0.6)
        .pointsData(samplePoints)
        .pointAltitude(0.01)
        .pointColor(() => "#ef4444")
        .pointRadius(0.7)

      globeRef.current = globe
      scene.add(globe)

      const ambient = new THREE.AmbientLight(0xffffff, 1.0)
      scene.add(ambient)

      containerRef.current!.appendChild(renderer.domElement)

      const onResize = () => {
        if (!containerRef.current) return
        const w = containerRef.current.clientWidth
        const h = 420
        renderer.setSize(w, h)
        camera.aspect = w / h
        camera.updateProjectionMatrix()
      }
      window.addEventListener("resize", onResize)

      const animate = () => {
        if (globeRef.current) globeRef.current.rotation.y += 0.0018
        renderer.render(scene, camera)
        animationId = requestAnimationFrame(animate)
      }
      animate()

      return () => {
        window.removeEventListener("resize", onResize)
        if (animationId) cancelAnimationFrame(animationId)
        if (renderer) {
          renderer.dispose()
          containerRef.current?.removeChild(renderer.domElement)
        }
      }
    }

    const cleanupPromise = setup()
    return () => {
      ;(async () => { const cleanup = await cleanupPromise; if (typeof cleanup === "function") cleanup() })()
    }
  }, [])

  return (
    <section id="globe" className="mx-auto w-full max-w-6xl mt-16">
      <div className="rounded-xl border border-white/10 bg-black/40 p-0 overflow-hidden">
        <div className="px-6 pt-6">
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-red-500" />
            <span>Global activity</span>
          </div>
          <h3 className="mt-2 text-xl md:text-2xl font-semibold">Federated rollouts across sites</h3>
          <p className="mt-1 text-gray-400">Red arcs simulate client updates travelling to the server. Points represent active sites.</p>
        </div>
        <div ref={containerRef} className="w-full" />
      </div>
    </section>
  )
}

interface LatLng {
  lat: number
  lng: number
}

function makeArc(from: LatLng, to: LatLng) {
  return { startLat: from.lat, startLng: from.lng, endLat: to.lat, endLng: to.lng }
}

const samplePoints: Array<LatLng> = [
  { lat: 37.7749, lng: -122.4194 }, // SF
  { lat: 40.7128, lng: -74.0060 },  // NYC
  { lat: 51.5074, lng: -0.1278 },   // London
  { lat: 48.8566, lng: 2.3522 },    // Paris
  { lat: 35.6895, lng: 139.6917 },  // Tokyo
]

const sampleArcs = [
  makeArc(samplePoints[0], samplePoints[1]),
  makeArc(samplePoints[1], samplePoints[2]),
  makeArc(samplePoints[2], samplePoints[4]),
  makeArc(samplePoints[4], samplePoints[0]),
]
