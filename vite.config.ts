import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/andres-net/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
