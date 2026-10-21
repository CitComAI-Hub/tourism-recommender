"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  type ReactNode,
} from "react";
import type { UserProfile, POI, PlannerPreferences } from "./api";

interface UserState {
  userId: string | null;
  profile: UserProfile | null;
  acceptedRecommendations: POI[];
  likedPOIs: string[];
  plannerPreferences: PlannerPreferences;
  onboardingComplete: boolean;
  isHydrated: boolean;
}

interface UserContextType extends UserState {
  setUserId: (id: string) => void;
  setProfile: (profile: UserProfile) => void;
  addAcceptedRecommendation: (poi: POI) => void;
  removeAcceptedRecommendation: (poiId: string) => void;
  clearAcceptedRecommendations: () => void;
  addLikedPOI: (poiId: string) => void;
  removeLikedPOI: (poiId: string) => void;
  setPlannerPreferences: (prefs: Partial<PlannerPreferences>) => void;
  completeOnboarding: () => void;
  resetUser: () => void;
}

const UserContext = createContext<UserContextType | undefined>(undefined);

const STORAGE_KEY = "valencia_tourism_user";

const DEFAULT_PLANNER_PREFS: PlannerPreferences = {
  max_stops_per_day: 5,
  max_total_minutes: 480,
  visit_time_minutes: 60,
  start_location: null,
  use_current_location: true,
  mobility: "walking",
};

const DEFAULT_STATE: UserState = {
  userId: null,
  profile: null,
  acceptedRecommendations: [],
  likedPOIs: [],
  plannerPreferences: DEFAULT_PLANNER_PREFS,
  onboardingComplete: false,
  isHydrated: false,
};

export function UserProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<UserState>(DEFAULT_STATE);

  // Load from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        setState({
          ...DEFAULT_STATE,
          ...parsed,
          isHydrated: true,
          likedPOIs: Array.isArray(parsed.likedPOIs) ? parsed.likedPOIs : [],
          plannerPreferences: {
            ...DEFAULT_PLANNER_PREFS,
            ...(parsed.plannerPreferences || {}),
          },
        });
      } catch {
        localStorage.removeItem(STORAGE_KEY);
        setState((prev) => ({ ...prev, isHydrated: true }));
      }
    } else {
      setState((prev) => ({ ...prev, isHydrated: true }));
    }
  }, []);

  // Save to localStorage on state change (even without userId to keep demo state)
  useEffect(() => {
    if (!state.isHydrated) return;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [state]);

  const setUserId = (id: string) => {
    setState((prev) => ({ ...prev, userId: id }));
  };

  const setProfile = (profile: UserProfile) => {
    setState((prev) => ({ ...prev, profile }));
  };

  const addAcceptedRecommendation = (poi: POI) => {
    setState((prev) => ({
      ...prev,
      acceptedRecommendations: prev.acceptedRecommendations.some(
        (p) => p.id === poi.id
      )
        ? prev.acceptedRecommendations
        : [...prev.acceptedRecommendations, poi],
    }));
  };

  const removeAcceptedRecommendation = (poiId: string) => {
    setState((prev) => ({
      ...prev,
      acceptedRecommendations: prev.acceptedRecommendations.filter(
        (p) => p.id !== poiId
      ),
    }));
  };

  const clearAcceptedRecommendations = () => {
    setState((prev) => ({ ...prev, acceptedRecommendations: [] }));
  };

  const addLikedPOI = (poiId: string) => {
    setState((prev) => ({
      ...prev,
      likedPOIs: prev.likedPOIs.includes(poiId)
        ? prev.likedPOIs
        : [...prev.likedPOIs, poiId],
    }));
  };

  const removeLikedPOI = (poiId: string) => {
    setState((prev) => ({
      ...prev,
      likedPOIs: prev.likedPOIs.filter((id) => id !== poiId),
    }));
  };

  const setPlannerPreferences = (prefs: Partial<PlannerPreferences>) => {
    setState((prev) => ({
      ...prev,
      plannerPreferences: { ...prev.plannerPreferences, ...prefs },
    }));
  };

  const completeOnboarding = () => {
    setState((prev) => ({ ...prev, onboardingComplete: true }));
  };

  const resetUser = () => {
    localStorage.removeItem(STORAGE_KEY);
    setState(DEFAULT_STATE);
  };

  return (
    <UserContext.Provider
      value={{
        ...state,
        setUserId,
        setProfile,
        addAcceptedRecommendation,
        removeAcceptedRecommendation,
        clearAcceptedRecommendations,
        addLikedPOI,
        removeLikedPOI,
        setPlannerPreferences,
        completeOnboarding,
        resetUser,
      }}
    >
      {children}
    </UserContext.Provider>
  );
}

export function useUser() {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error("useUser must be used within a UserProvider");
  }
  return context;
}
