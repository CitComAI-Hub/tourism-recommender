import React from "react"
import type { Metadata, Viewport } from "next";
import { DM_Sans, DM_Serif_Display } from "next/font/google";
import { UserProvider } from "@/lib/user-context";
import { I18nProvider } from "@/lib/i18n";
import "leaflet/dist/leaflet.css";
import "./globals.css";

const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";

const dmSans = DM_Sans({ subsets: ["latin"], variable: "--font-sans" });
const dmSerif = DM_Serif_Display({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-serif",
});

export const metadata: Metadata = {
  title: "Valencia Explorer - Descubre Valencia a tu manera",
  description:
    "Tu guia personalizada para explorar Valencia. Recomendaciones unicas basadas en tus preferencias y planifica tu ruta perfecta.",
  icons: {
    icon: [
      {
        url: `${BASE}/icon-light-32x32.png`,
        media: "(prefers-color-scheme: light)",
      },
      {
        url: `${BASE}/icon-dark-32x32.png`,
        media: "(prefers-color-scheme: dark)",
      },
      {
        url: `${BASE}/icon.svg`,
        type: "image/svg+xml",
      },
    ],
    apple: `${BASE}/apple-icon.png`,
  },
};

export const viewport: Viewport = {
  themeColor: "#2d6a4f",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="es">
      <body
        className={`${dmSans.variable} ${dmSerif.variable} font-sans antialiased`}
      >
        <I18nProvider>
          <UserProvider>{children}</UserProvider>
        </I18nProvider>
      </body>
    </html>
  );
}
