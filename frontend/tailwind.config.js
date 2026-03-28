/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        bloomberg: {
          orange: '#FF5F00',
          dark: '#010B13',
          panel: '#05121F',
          border: '#1e293b'
        }
      }
    },
  },
  plugins: [],
}