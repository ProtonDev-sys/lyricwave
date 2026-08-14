import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { headers } from "next/headers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host") ?? requestHeaders.get("host") ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto") ?? (host.startsWith("localhost") ? "http" : "https");
  const origin = `${protocol}://${host}`;

  return {
    title: "lyricwave — Your song, word for word",
    description:
      "Isolate vocals and create word-by-word timed live lyrics with a private local GPU engine.",
    applicationName: "lyricwave",
    icons: {
      icon: "/favicon.png",
      shortcut: "/favicon.png",
    },
    openGraph: {
      type: "website",
      title: "lyricwave — Your song, word for word",
      description: "Private, local-GPU vocal isolation and word-synced live lyrics.",
      url: origin,
      siteName: "lyricwave",
      images: [{ url: `${origin}/og.png`, width: 1536, height: 910, alt: "lyricwave live lyrics" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "lyricwave — Your song, word for word",
      description: "Private, local-GPU vocal isolation and word-synced live lyrics.",
      images: [`${origin}/og.png`],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
