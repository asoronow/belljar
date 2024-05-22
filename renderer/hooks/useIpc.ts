// hooks/useIpc.ts
import { useEffect } from "react";

export const useIpc = (channel: string, callback: (...args: any[]) => void) => {
  useEffect(() => {
    const listener = (...args: any[]) => {
      callback(...args);
    };

    window.ipc.on(channel, listener);
  }, [channel, callback]);
};
