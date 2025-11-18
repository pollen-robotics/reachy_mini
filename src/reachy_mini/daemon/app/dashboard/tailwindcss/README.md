# Generate static Tailwind CSS

## Requirements

(From this folder)

```bash
pnpm install -D tailwindcss @tailwindcss/cli 
```

## Generate CSS

```bash
npx @tailwindcss/cli -i ./styles/app.css -o ../static/css/app.css
```