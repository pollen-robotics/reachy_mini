import { defineConfig } from "allure";

export default defineConfig({
  name: "Reachy Mini Tests",
  output: "./allure-report",
  historyPath: "./history/history.jsonl",
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
