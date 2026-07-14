import { spawn } from "child_process";

const port = process.env.PORT || "3000";
console.log(`Starting Streamlit dashboard on port ${port}...`);

const streamlit = spawn(
  "streamlit",
  [
    "run",
    "app.py",
    "--server.port",
    port,
    "--server.address",
    "0.0.0.0",
    "--server.headless",
    "true",
    "--server.enableCORS",
    "false",
    "--server.enableXsrfProtection",
    "false",
    "--server.enableWebsocketCompression",
    "false",
    "--browser.gatherUsageStats",
    "false"
  ],
  {
    stdio: "inherit",
    shell: true,
  }
);

streamlit.on("close", (code) => {
  console.log(`Streamlit process exited with code ${code}`);
  process.exit(code || 0);
});
