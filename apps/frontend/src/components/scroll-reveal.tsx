"use client"

import { useEffect } from "react"

export function ScrollReveal() {
  useEffect(() => {
    if (!("IntersectionObserver" in window)) {
      document.querySelectorAll<HTMLElement>(".reveal").forEach(el => el.classList.add("is-visible"))
      return
    }
    const io = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        const el = entry.target as HTMLElement
        if (entry.isIntersecting) el.classList.add("is-visible")
      })
    }, { rootMargin: "-10% 0px -10% 0px", threshold: 0.05 })

    document.querySelectorAll<HTMLElement>(".reveal").forEach(el => io.observe(el))

    return () => io.disconnect()
  }, [])

  return null
}


