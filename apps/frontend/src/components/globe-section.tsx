"use client";

import { useEffect, useRef } from "react"

export function GlobeSection() {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const globeRef = useRef<unknown>(null)

  useEffect(() => {
    let renderer: unknown
    let scene: unknown
    let camera: unknown
    let animationId: number | undefined

    async function setup() {
      const THREE = await import("three")
      const Globe = (await import("three-globe")).default

      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
      renderer.setSize(containerRef.current!.clientWidth, 560)
      renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1))

      scene = new THREE.Scene()
      camera = new THREE.PerspectiveCamera(55, (containerRef.current!.clientWidth) / 560, 0.1, 1000)
      camera.position.z = 280

      const globe = new Globe()
        .globeImageUrl("//unpkg.com/three-globe/example/img/earth-night.jpg")
        .bumpImageUrl("//unpkg.com/three-globe/example/img/earth-topology.png")
        .showAtmosphere(true)
        .atmosphereColor("#3b82f6")
        .atmosphereAltitude(0.12)
        // clients → one green server
        .arcsData(arcsToServer)
        .arcColor(() => ["#ef4444", "#22c55e"]) // red to green
        .arcAltitude(() => Math.random() * 0.25 + 0.08)
        .arcStroke(() => 0.55)
        .arcDashLength(0.5)
        .arcDashGap(0.7)
        .arcDashAnimateTime(2000)
        .pointsData(pointsWithServer)
        .pointAltitude((p: { isServer?: boolean }) => p.isServer ? 0.025 : 0.012)
        .pointColor((p: { isServer?: boolean }) => p.isServer ? "#22c55e" : "#ef4444")
        .pointRadius((p: { isServer?: boolean }) => p.isServer ? 1.3 : 0.7)

      globeRef.current = globe
      scene.add(globe)

      const ambient = new THREE.AmbientLight(0xffffff, 1.1)
      scene.add(ambient)

      // Countries outline for visibility
      try {
        const res = await fetch("//unpkg.com/three-globe/example/datasets/ne_110m_admin_0_countries.geojson")
        const geo = await res.json()
        globe
          .polygonsData(geo.features)
          .polygonCapColor(() => "rgba(255,255,255,0.02)")
          .polygonSideColor(() => "rgba(255,255,255,0.08)")
          .polygonStrokeColor(() => "rgba(255,255,255,0.4)")
          .polygonAltitude(0.003)
      } catch {}

      containerRef.current!.appendChild(renderer.domElement)

      const onResize = () => {
        if (!containerRef.current) return
        const w = containerRef.current.clientWidth
        const h = 560
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
          <h3 className="mt-2 text-xl md:text-2xl font-semibold">Clients to server, worldwide</h3>
          <p className="mt-1 text-gray-400">Many red clients stream updates to a single green server. Countries outlined, bigger globe.</p>
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

// Many red client points across the world
const clientPoints: Array<LatLng> = [
  { lat: 37.7749, lng: -122.4194 }, { lat: 34.0522, lng: -118.2437 }, { lat: 47.6062, lng: -122.3321 },
  { lat: 40.7128, lng: -74.0060 }, { lat: 25.7617, lng: -80.1918 }, { lat: 19.4326, lng: -99.1332 },
  { lat: 51.5074, lng: -0.1278 }, { lat: 52.52, lng: 13.405 }, { lat: 48.8566, lng: 2.3522 },
  { lat: 41.9028, lng: 12.4964 }, { lat: 59.3293, lng: 18.0686 }, { lat: 55.7558, lng: 37.6173 },
  { lat: 35.6895, lng: 139.6917 }, { lat: 37.5665, lng: 126.9780 }, { lat: 31.2304, lng: 121.4737 },
  { lat: 22.3193, lng: 114.1694 }, { lat: 1.3521, lng: 103.8198 }, { lat: -33.8688, lng: 151.2093 },
  { lat: -37.8136, lng: 144.9631 }, { lat: -23.5505, lng: -46.6333 }, { lat: -34.6037, lng: -58.3816 },
  { lat: 6.5244, lng: 3.3792 }, { lat: -1.2921, lng: 36.8219 }, { lat: 30.0444, lng: 31.2357 },
]

// Single green server (e.g., Frankfurt)
const server: LatLng & { isServer?: boolean } = { lat: 50.1109, lng: 8.6821 }

const pointsWithServer = [...clientPoints.map(p => ({ ...p, isServer: false })), { ...server, isServer: true }]

const arcsToServer = clientPoints.map(p => makeArc(p, server))
