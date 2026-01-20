/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Provider-specific colors
        provider: {
          amazon: '#FF9900',
          anthropic: '#D97706',
          meta: '#0668E1',
          mistral: '#F97316',
          cohere: '#39A7FF',
          ai21: '#6366F1',
          stability: '#8B5CF6',
        },
      },
    },
  },
  plugins: [],
}
