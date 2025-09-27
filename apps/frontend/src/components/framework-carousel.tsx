'use client'

import { useState, useEffect } from 'react'
import Image from 'next/image'

const frameworks = [
  { name: 'PyTorch', logo: '/pytorch-logo-color.png', alt: 'PyTorch' },
  { name: 'TensorFlow', logo: '/tensorflow-logo-nobg.svg', alt: 'TensorFlow' },
  { name: 'JAX', logo: '/jax-logo-color.png', alt: 'JAX' },
  { name: 'scikit-learn', logo: '/sklearn-logo-wiki.svg', alt: 'scikit-learn' },
  { name: 'Pandas', logo: '/pandas-logo-color.svg', alt: 'Pandas' }
]

export function FrameworkCarousel() {
  return (
    <section className="mx-auto w-full max-w-4xl mt-12">
      <h2 className="text-2xl md:text-3xl font-semibold text-center mb-8">Compatible Frameworks</h2>

      <div className="relative overflow-hidden rounded-lg border border-white/10 bg-gray-950 p-8 pb-12">
        <div className="flex space-x-16 animate-infinite-scroll">
          {[...frameworks, ...frameworks, ...frameworks, ...frameworks].map((framework, index) => (
            <div key={`${framework.name}-${index}`} className="flex-shrink-0 flex flex-col items-center justify-center min-w-[280px]">
              <div className="relative w-44 h-44 mb-6 flex items-center justify-center p-4">
                <Image
                  src={framework.logo}
                  alt={framework.alt}
                  fill
                  className="object-contain opacity-90 hover:opacity-100 transition-opacity"
                />
              </div>
              <h3 className="text-lg font-medium text-gray-300">{framework.name}</h3>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}