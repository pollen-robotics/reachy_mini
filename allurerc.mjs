import { defineConfig } from "allure";

export default defineConfig({
  name: "Reachy Mini Tests",
  output: "./allure-report",
  historyPath: "./history/history.jsonl",
  // ~0.7 MB per run: unbounded, the file outgrew GitHub's 100 MB limit.
  historyLimit: 30,
  plugins: {
    awesome: {
      import: "@allurereport/plugin-awesome",
      options: {
        reportName: "Reachy Mini Tests",
        singleFile: false,
      },
    },
  },
});
