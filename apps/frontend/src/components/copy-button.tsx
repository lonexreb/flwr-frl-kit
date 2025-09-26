"use client";

import { useState } from "react";

export interface CopyButtonProps {
  text: string
  label?: string
}

export function CopyButton({ text, label }: CopyButtonProps) {
  const [isCopied, setIsCopied] = useState(false)

  async function handleCopy() {
    if (!text) return
    try {
      await navigator.clipboard.writeText(text)
      setIsCopied(true)
      setTimeout(() => setIsCopied(false), 1500)
    } catch {
      setIsCopied(false)
    }
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="inline-flex items-center gap-2 rounded-md border border-black/10 dark:border-white/10 px-3 py-1.5 text-sm hover:bg-black/5 dark:hover:bg-white/10 transition-colors"
      aria-label={label ?? "Copy to clipboard"}
    >
      <span className="font-medium">{isCopied ? "Copied" : label ?? "Copy"}</span>
      <svg
        xmlns="http://www.w3.org/2000/svg"
        viewBox="0 0 24 24"
        fill="currentColor"
        className="h-4 w-4"
        aria-hidden
      >
        <path d="M16 1H4a2 2 0 0 0-2 2v12h2V3h12V1Zm3 4H8a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2Zm0 14H8V7h11v12Z" />
      </svg>
    </button>
  )
}


