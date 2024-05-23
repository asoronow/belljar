// hooks/useIpcListener.ts
import { useEffect } from "react";

export const useIpcListener = (
  channel: string,
  callback: (...args: unknown[]) => void
) => {
  useEffect(() => {
    const unsubscribe = window.ipc.on(channel, callback);

    return () => {
      unsubscribe();
    };
  }, [channel, callback]);
};
