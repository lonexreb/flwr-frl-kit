"use client";

import { useState } from "react";
import Image from "next/image";

export function DemoGif() {
  const [showPlaceholder, setShowPlaceholder] = useState(false);

  return (
    <div className="aspect-video bg-black/60 rounded-lg border border-white/10 flex items-center justify-center">
      <Image 
        src="/3x.gif" 
        alt="FRL Kit Demo" 
        width={800}
        height={450}
        className="max-w-full max-h-full rounded-lg"
        onError={() => setShowPlaceholder(true)}
        style={{ display: showPlaceholder ? 'none' : 'block' }}
      />
      <div 
        className="flex-col items-center justify-center text-gray-500 text-sm"
        style={{ display: showPlaceholder ? 'flex' : 'none' }}
      >
        <svg className="w-12 h-12 mb-2" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
        </svg>
        <span>Demo GIF placeholder</span>
        <span className="text-xs mt-1">Replace /demo.gif to show your demo</span>
      </div>
    </div>
  );
}
