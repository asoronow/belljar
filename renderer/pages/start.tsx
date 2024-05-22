import React, { useState } from "react";
import Head from "next/head";
import { useIpc } from "../hooks/useIpc";
import { ScaleLoader } from "react-spinners";
export default function HomePage() {
  const [message, setMessage] = useState("Getting things ready...");
  const [loadProgress, setLoadProgress] = useState(0);

  useIpc("setup-progress", (message) => {
    setMessage(message);
  });

  return (
    <React.Fragment>
      <Head>
        <title>Bell Jar - Setting Things Up</title>
      </Head>
      <div className="flex flex-col items-center justify-center min-h-screen py-6 bg-white text-center">
        <h1 className="text-4xl font-bold text-black">Bell Jar</h1>
        <h2 className="text-2xl font-semibold text-gray-600">v10.0.0</h2>
        <p className="text-lg text-gray-500 my-4">{message}</p>
        <ScaleLoader height={15} color="#000" />
      </div>
    </React.Fragment>
  );
}
