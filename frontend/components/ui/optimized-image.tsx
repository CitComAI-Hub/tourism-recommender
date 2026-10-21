"use client";

import Image, { type ImageProps } from "next/image";
import { useState, useMemo } from "react";
import { cn } from "@/lib/utils";
import { ImageOff } from "lucide-react";

/**
 * Convert Wikimedia Commons `Special:FilePath/...?width=N` redirect URLs
 * into direct `upload.wikimedia.org` thumbnail URLs that don't require
 * a 302 redirect (which Next.js image optimiser handles poorly).
 *
 * Other URLs are returned unchanged.
 */
export function normalizeImageUrl(url: string): string {
    try {
        const parsed = new URL(url);

        // Match: commons.wikimedia.org/wiki/Special:FilePath/<filename>?width=<N>
        if (
            parsed.hostname === "commons.wikimedia.org" &&
            parsed.pathname.startsWith("/wiki/Special:FilePath/")
        ) {
            // Extract the raw filename (already percent-encoded in the URL path)
            const encodedFilename = parsed.pathname.replace(
                "/wiki/Special:FilePath/",
                ""
            );
            const width = parsed.searchParams.get("width") || "800";

            // Build the direct thumbnail URL
            // upload.wikimedia.org uses MD5-based subfolders, but the /thumb/ endpoint
            // with Special:FilePath semantics also works when routed through the
            // commons.wikimedia.org/w/thumb.php endpoint:
            return `https://commons.wikimedia.org/w/thumb.php?f=${encodedFilename}&w=${width}`;
        }
    } catch {
        // If URL parsing fails, return as-is
    }
    return url;
}

interface OptimizedImageProps extends Omit<ImageProps, "onLoad" | "onError"> {
    /** Extra classes for the wrapper div (should be positioned + have dimensions) */
    wrapperClassName?: string;
    /** Icon size for the error fallback (default 24) */
    fallbackIconSize?: number;
}

/**
 * Wrapper around next/image that adds:
 * - Animated shimmer gradient while loading
 * - Smooth fade-in transition on load
 * - Error fallback with icon
 * - Automatic normalisation of Wikimedia redirect URLs
 */
export function OptimizedImage({
    wrapperClassName,
    className,
    fallbackIconSize = 24,
    alt,
    src,
    ...props
}: OptimizedImageProps) {
    const [loaded, setLoaded] = useState(false);
    const [errored, setErrored] = useState(false);

    // Normalise Wikimedia redirect URLs to direct ones
    const resolvedSrc = useMemo(() => {
        if (typeof src === "string") return normalizeImageUrl(src);
        return src;
    }, [src]);

    if (errored || !resolvedSrc) {
        return (
            <div
                className={cn(
                    "flex items-center justify-center bg-muted",
                    wrapperClassName
                )}
            >
                <ImageOff
                    className="text-muted-foreground/40"
                    style={{ width: fallbackIconSize, height: fallbackIconSize }}
                />
            </div>
        );
    }

    return (
        <div className={cn("overflow-hidden relative", wrapperClassName)}>
            {/* Shimmer skeleton – visible until image loads */}
            {!loaded && (
                <div className="absolute inset-0 bg-muted">
                    <div
                        className="absolute inset-0 animate-shimmer"
                        style={{
                            background:
                                "linear-gradient(90deg, transparent 0%, hsl(var(--muted-foreground) / 0.08) 40%, hsl(var(--muted-foreground) / 0.15) 50%, hsl(var(--muted-foreground) / 0.08) 60%, transparent 100%)",
                            backgroundSize: "200% 100%",
                        }}
                    />
                </div>
            )}

            <Image
                alt={alt}
                src={resolvedSrc}
                className={cn(
                    "transition-opacity duration-500 ease-out",
                    loaded ? "opacity-100" : "opacity-0",
                    className
                )}
                onLoad={() => setLoaded(true)}
                onError={() => setErrored(true)}
                {...props}
            />
        </div>
    );
}
