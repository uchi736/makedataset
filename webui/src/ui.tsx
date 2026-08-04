import { createContext, useCallback, useContext, useRef, useState } from "react";
import type { ReactNode } from "react";

// ---------------- ツールチップ (全チャート共通のホバー層) ----------------

interface TipState { x: number; y: number; content: ReactNode }

interface TipApi {
  show: (e: { clientX: number; clientY: number }, content: ReactNode) => void;
  hide: () => void;
}

const TipContext = createContext<TipApi>({ show: () => {}, hide: () => {} });

export function TooltipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const raf = useRef(0);

  const show = useCallback((e: { clientX: number; clientY: number }, content: ReactNode) => {
    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(() => {
      const x = Math.min(e.clientX + 14, window.innerWidth - 300);
      const y = Math.min(e.clientY + 14, window.innerHeight - 90);
      setTip({ x, y, content });
    });
  }, []);

  const hide = useCallback(() => {
    cancelAnimationFrame(raf.current);
    setTip(null);
  }, []);

  return (
    <TipContext.Provider value={{ show, hide }}>
      {children}
      {tip && (
        <div className="tooltip" style={{ left: tip.x, top: tip.y }}>
          {tip.content}
        </div>
      )}
    </TipContext.Provider>
  );
}

export function useTooltip(): TipApi {
  return useContext(TipContext);
}

// ---------------- テーマ (light/dark) ----------------

export const ThemeContext = createContext<{ dark: boolean }>({ dark: false });

export function useDark(): boolean {
  return useContext(ThemeContext).dark;
}
