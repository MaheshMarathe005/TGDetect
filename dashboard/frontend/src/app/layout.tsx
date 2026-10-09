import type { Metadata } from "next";

import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { ThemeProvider } from "@/lib/theme-context";
import { ModelProvider } from "@/lib/model-context";
import { DatasetProvider } from "@/lib/dataset-context";

export const metadata: Metadata = {
  title: "TGDetect — Temporal Graph Neural Network for Threat Detection",
  description:
    "Backend-contract-ready frontend for the TGDetect backend: TGEvent investigation, temporal heterogeneous graph, attack-chain reconstruction, parquet artifacts, and GAT and causal node-memory model model views.",
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
        <ThemeProvider><ModelProvider><DatasetProvider>{children}</DatasetProvider></ModelProvider></ThemeProvider>
        <Toaster />
      </body>
    </html>
  );
}
