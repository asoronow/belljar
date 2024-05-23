// hooks/useIpc.ts
import { useCallback } from "react";

export const useIpc = () => {
  const send = useCallback((channel: string, value: unknown) => {
    window.ipc.send(channel, value);
  }, []);

  return { send };
};
