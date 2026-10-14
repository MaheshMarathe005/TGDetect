import type { Metadata } from "next";

import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { ThemeProvider } from "@/lib/theme-context";

export const metadata: Metadata = {
  title: "TGDetect — Temporal Graph Neural Network for Threat Detection",
  description:
    "CADETS temporal graph investigation with the trained V3 detector, traceable alert evidence and candidate attack-chain reconstruction.",
  keywords: [
    "TGDetect",
    "Temporal Graph",
    "Graph Neural Network",
    "GAT",
    "Node memory",
    "TGNN",
    "Threat Detection",
    "Attack Chain",
    "MITRE ATT&CK",
  ],
  authors: [{ name: "TGDetect" }],
  icons: { icon: "/logo.svg" },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body
        className={`antialiased bg-background text-foreground`}
      >
        <ThemeProvider>{children}</ThemeProvider>
        <Toaster />
      </body>
    </html>
  );
}
