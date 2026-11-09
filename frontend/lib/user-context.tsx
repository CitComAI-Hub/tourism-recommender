"use client";

import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import type { Location, UserProfile } from "./api";

/** Everything about the user lives in this browser only; the API is stateless. */
interface UserState {
  profile: UserProfile | null;
  likedPOIs: string[];
  dislikedPOIs: string[];
  startLocation: Location | null;
  useCurrentLocation: boolean;
  onboardingComplete: boolean;
  isHydrated: boolean;
}

interface UserContextType extends UserState {
  setProfile: (profile: UserProfile) => void;
  setFeedback: (liked: string[], disliked: string[]) => void;
  like: (poiId: string) => void;
  dislike: (poiId: string) => void;
  clearFeedback: (poiId: string) => void;
  setLocationPreference: (startLocation: Location | null, useCurrentLocation: boolean) => void;
  completeOnboarding: () => void;
  resetUser: () => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

const STORAGE_KEY = "tourism_user_v2";

const DEFAULT_STATE: UserState = {
  profile: null,
  likedPOIs: [],
  dislikedPOIs: [],
  startLocation: null,
  useCurrentLocation: true,
  onboardingComplete: false,
  isHydrated: false,
};

const without = (list: string[], id: string) => list.filter((x) => x !== id);

export function UserProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<UserState>(DEFAULT_STATE);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      const parsed = stored ? JSON.parse(stored) : {};
      setState({
        ...DEFAULT_STATE,
        ...parsed,
        likedPOIs: Array.isArray(parsed.likedPOIs) ? parsed.likedPOIs : [],
        dislikedPOIs: Array.isArray(parsed.dislikedPOIs) ? parsed.dislikedPOIs : [],
        isHydrated: true,
      });
    } catch {
      setState({ ...DEFAULT_STATE, isHydrated: true });
    }
  }, []);

  useEffect(() => {
    if (!state.isHydrated) return;
    try {
      const { isHydrated: _, ...persisted } = state;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(persisted));
    } catch {
      // Private mode or blocked storage: keep working in memory.
    }
  }, [state]);

  const value: UserContextType = {
    ...state,
    setProfile: (profile) => setState((s) => ({ ...s, profile })),
    setFeedback: (liked, disliked) => setState((s) => ({ ...s, likedPOIs: liked, dislikedPOIs: disliked })),
    like: (id) =>
      setState((s) => ({
        ...s,
        likedPOIs: s.likedPOIs.includes(id) ? s.likedPOIs : [...s.likedPOIs, id],
        dislikedPOIs: without(s.dislikedPOIs, id),
      })),
    dislike: (id) =>
      setState((s) => ({
        ...s,
        dislikedPOIs: s.dislikedPOIs.includes(id) ? s.dislikedPOIs : [...s.dislikedPOIs, id],
        likedPOIs: without(s.likedPOIs, id),
      })),
    clearFeedback: (id) =>
      setState((s) => ({ ...s, likedPOIs: without(s.likedPOIs, id), dislikedPOIs: without(s.dislikedPOIs, id) })),
    setLocationPreference: (startLocation, useCurrentLocation) =>
      setState((s) => ({ ...s, startLocation, useCurrentLocation })),
    completeOnboarding: () => setState((s) => ({ ...s, onboardingComplete: true })),
    resetUser: () => {
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch {
        // ignore
      }
      setState({ ...DEFAULT_STATE, isHydrated: true });
    },
  };

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
}

export function useUser() {
  const context = useContext(UserContext);
  if (context === undefined) throw new Error("useUser must be used within a UserProvider");
  return context;
}
