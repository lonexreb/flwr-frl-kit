import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "FLWR FRL Kit",
  description: "Minimal landing page",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-black text-gray-200`}>
        <header className="sticky top-0 z-10 border-b border-white/10 bg-black/60 backdrop-blur">
          <nav className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between px-4">
            <div className="flex items-center gap-2">
              <span className="inline-block h-2 w-2 rounded-full bg-red-500" />
              <span className="text-sm tracking-wider">flwr-frl-kit</span>
            </div>
            <div className="flex items-center gap-4 text-sm">
              <a className="hover:underline" href="#mission">Mission</a>
              <a className="btn-primary rounded-md px-3 py-1.5 text-sm font-medium" href="#beta">Get Beta Access</a>
            </div>
          </nav>
        </header>
        {children}
      </body>
    </html>
  );
}
