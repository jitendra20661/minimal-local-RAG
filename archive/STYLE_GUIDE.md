# Perplexity Pro Style Guide

## Overview
This style guide documents the design system for a dark-themed AI search interface inspired by Perplexity Pro.

---

## Color System

### Theme
**Dark Mode** (Primary)
- Color scheme: Dark with high contrast
- Font weight adjustment: Lighter weights (375 instead of 400 for normal)
- Letter spacing: Increased to `0.01em` for better readability

### Color Variables (OKLCH Format)

#### Background Colors
```css
--background-underlay-color: /* Deepest background layer */
--background-base-color: #100e12 /* Primary dark background */
--background-subtle-color: /* Slightly lighter than base */
--background-subtler-color: /* More prominent surfaces */
--background-subtlest-color: /* Lightest background variant */
--background-raised-color: /* Elevated UI elements */
--background-elevated-color: /* Modal/overlay backgrounds */
--background-inverse-color: /* Light mode equivalent */
```

#### Foreground/Text Colors
```css
--foreground-color: /* Primary text color */
--foreground-quiet-color: /* Secondary text */
--foreground-quieter-color: /* Tertiary text */
--foreground-quietest-color: /* Most subtle text */
--foreground-subtle-color: /* De-emphasized text */
--foreground-subtler-color: /* More de-emphasized */
--foreground-subtlest-color: /* Least prominent text */
--foreground-inverse-color: /* Dark text for light backgrounds */
```

#### Accent Colors
```css
--super-color: /* Primary brand color */
--super-bg-color: /* Primary brand background */
--max-color: /* Maximum emphasis */
--offset-color: /* Visual separation */
--offset-plus-color: /* Enhanced separation */
--raised-offset-color: /* Elevated separation */
```

#### Semantic Colors
```css
--caution-color: /* Warning states */
--attention-color: /* Requires attention */
--positive-color: /* Success states */
--negative-color: /* Error states */
```

#### UI Colors
```css
--border-color: /* Border/divider color */
--backdrop-color: /* Modal backdrop overlay */
--shadow-overlay-border: rgba(255, 255, 255, .1) /* Subtle overlay borders */
```

### Specific Color Palette (OKLCH)
```css
/* Yellows */
--pale-yellow-50: 99.62% .004 106.47
--pale-yellow-100: 99.02% .004 106.47
--pale-yellow-200: 96.28% .007 106.52
--pale-yellow-300: 92.96% .007 106.53
--pale-yellow-600: 88.28% .012 106.65
--pale-yellow-800: 68.98% .027 109.55

/* Teals/Cyans */
--pale-teal-100: 96.95% .014 196.93
--mint-50: 98.85% .012 170.28
--mint-150: 93.8% .015 171.04
--pale-cyan-50: 49.33% .019 171.99
--hydra-150: 94.94% .033 208.37
--hydra-350: 71.92% .112 205.51
--hydra-450: 55.27% .086 208.61

/* Reds */
--red-100: 53.47% .151 25.99
--red-200: 51.83% .168 21.78
```

---

## Typography

### Font Families
```css
--font-fk-grotesk-neue: "fkGroteskNeue" /* Primary interface font */
--font-fk-grotesk: "fkGrotesk" /* Alternative grotesque */
--font-berkeley-mono: "berkeleyMono" /* Monospace/code */
--font-pp-editorial: "ppEditorial" /* Editorial content */
--font-instrument-serif: "instrumentSerif" /* Serif accents */
```

**Font Stack:**
```css
font-family: var(--font-fk-grotesk-neue), ui-sans-serif, system-ui,
             -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
             Helvetica Neue, Arial, "Noto Sans", sans-serif;
```

### Font Weights (Dark Mode Adjusted)
```css
--font-thin: 75
--font-extralight: 175
--font-light: 275
--font-normal: 375 /* Default body text */
--font-semimedium: 450
--font-medium: 475
--font-semibold: 575
--font-bold: 675
--font-extrabold: 775
--font-black: 875
```

### Typography Settings
```css
font-size: 16px /* Base font size */
line-height: 1.5
letter-spacing: 0.005em /* Light mode */
letter-spacing: 0.01em /* Dark mode - enhanced readability */
font-weight: 375 /* Dark mode normal weight */
```

---

## Spacing System

### Size Scale
```css
--size-2xs: 2px
--size-xs: 4px
--size-sm: 8px
--size-md: 16px
--size-ml: 24px
--size-lg: 32px
--size-xl: 48px
```

### Layout Dimensions
```css
/* Header & Navigation */
--header-height: 54px
--sidebar-width: 220px
--sidebar-width-collapsed: 90px
--sidebar-default-width: 72px
--sidebar-pinned-width: calc(200px + var(--sidebar-default-width))

/* Content Areas */
--thread-width: 1100px
--thread-content-width: 740px
--page-horizontal-padding: var(--size-md) /* 16px */

/* Heights */
--page-content-height: calc(100dvh - var(--header-height))
--page-content-height-without-header: 100dvh
--thread-input-height-with-padding: 130px
--thread-attachments-height-with-padding: 182px

/* Mobile */
--mobile-nav-height: env(safe-area-inset-bottom, 0)
--safe-area-inset-bottom: env(safe-area-inset-bottom, 0)
--min-touch-target: 2.75rem /* 44px - accessibility */
```

---

## Components

### Primary Search Input
**Visual Characteristics:**
- Dark background with subtle border
- Rounded corners (likely 12-16px based on interface)
- Placeholder: "Ask anything. Type @ for mentions."
- Light gray placeholder text color

**Input Container:**
- Max width: `var(--thread-width)` (1100px)
- Content width: `var(--thread-content-width)` (740px)
- Padding: Generous internal spacing
- Border: Subtle, uses `--border-color`

**Icon Toolbar:**
- Left side: Search, attachment, location icons
- Right side: Globe, media, attachment, mic, voice input
- Icon color: Cyan/teal accent (`--hydra-*` colors)
- Icon spacing: 8-16px apart

### Suggestion Pills
**Quick Actions:**
- Health, Teach me, Perplexity 101, Summarize, Get a job
- Background: `--background-subtle-color`
- Border: `--border-color`
- Border radius: Fully rounded (pill shape)
- Padding: 8px 16px
- Icon + text layout
- Hover state: Lighter background

### Logo
**Perplexity Pro:**
- Two-tone: "perplexity" in white/light, "pro" in cyan
- Font: Custom/geometric sans-serif
- Accent color: `--hydra-*` cyan for "pro"
- Position: Centered at top

### Sidebar Navigation
**Structure:**
- Fixed left position
- Width: 72px (collapsed state shown)
- Icons: Vertically stacked
- Items: Home, Discover, Spaces, Finance, etc.

**Icon Styling:**
- Size: ~24px
- Color: `--foreground-subtle-color`
- Active state: Cyan accent
- Spacing: Equal vertical distribution

**Account Section:**
- Bottom-aligned
- Pro badge: Cyan background
- Avatar: Circular with letter "M"
- Notifications: Red dot indicator

---

## Interactions

### Hover States
```css
/* Buttons/Pills */
background: Lighter than base state
transition: background 150ms ease

/* Icons */
color: Brighten or shift to accent color
```

### Focus States
```css
/* Form inputs */
outline: 2px solid var(--hydra-350)
outline-offset: 2px
```

### Active States
```css
/* Selected items */
background: var(--background-raised-color)
color: var(--super-color) /* Cyan accent */
```

---

## Shadows & Borders

### Shadows
```css
--shadow-overlay-border: rgba(255, 255, 255, .1) /* Subtle top borders */
```

### Border Radius
```css
/* Common values (inferred from design): */
- Small elements: 8px
- Medium (cards): 12px
- Large (modals): 16px
- Pills/buttons: 9999px (fully rounded)
```

### Border Colors
```css
border-color: oklch(var(--foreground-subtler-color))
/* or */
border-color: var(--border-color)
```

---

## Scrollbars

```css
scrollbar-width: 15px
scrollbar-color: initial /* Browser default for dark mode */
```

---

## Accessibility

### Touch Targets
```css
--min-touch-target: 2.75rem /* 44px minimum */
```

### Safe Areas (Mobile)
```css
--safe-area-inset-bottom: env(safe-area-inset-bottom, 0)
padding-bottom: var(--safe-area-inset-bottom)
```

### Color Contrast
- Ensure text meets WCAG AA standards
- Dark backgrounds with light text
- Accent colors (cyan) have sufficient contrast

---

## Dark Mode Specifications

### Font Weight Adjustments
Dark mode uses lighter font weights for better rendering on dark backgrounds:
- Normal: 375 (vs 400 in light mode)
- All weights reduced by 25 units

### Letter Spacing
- Light mode: `0.005em`
- Dark mode: `0.01em` (increased for readability)

### Background
```css
html[data-color-scheme=dark] {
    background-color: #100e12 !important;
}
```

---

## Theming System

### CSS Custom Properties
All colors use CSS custom properties for easy theming:

```css
[data-color-scheme=dark] {
    --foreground-color: var(--dark-foreground-color);
    /* ... */
}

[data-color-scheme=light] {
    --foreground-color: var(--light-foreground-color);
    /* ... */
}
```

### Prefers Color Scheme
Automatic theme detection:
```css
@media (prefers-color-scheme: dark) {
    :root:not([data-color-scheme=light]) {
        /* Apply dark theme */
    }
}
```

---

## Animation & Transitions

### CSS Variables for Transforms
```css
--tw-translate-x: 0
--tw-translate-y: 0
--tw-rotate: 0
--tw-scale-x: 1
--tw-scale-y: 1
```

### Smooth Scrolling
```css
html {
    scroll-behavior: smooth;
    overflow-y: auto;
}
```

---

## Usage Guidelines

### Do's
✓ Use semantic color variables (`--foreground-color`, not hardcoded values)
✓ Maintain minimum touch target of 44px
✓ Use spacing system variables for consistent gaps
✓ Apply dark mode font weight adjustments
✓ Respect safe area insets on mobile

### Don'ts
✗ Don't hardcode color values
✗ Don't use black (#000000) - use `--background-base-color`
✗ Don't ignore touch target sizes
✗ Don't skip hover/focus states
✗ Don't use fixed pixel values for spacing (use variables)

---

## Implementation Example

```html
<div class="search-container">
    <div class="search-input-wrapper">
        <input
            type="text"
            placeholder="Ask anything. Type @ for mentions."
            class="search-input"
        />
        <div class="icon-toolbar">
            <button class="icon-button">
                <svg><!-- Search icon --></svg>
            </button>
            <!-- More icons -->
        </div>
    </div>

    <div class="suggestion-pills">
        <button class="pill">
            <svg><!-- Icon --></svg>
            <span>Health</span>
        </button>
        <!-- More pills -->
    </div>
</div>
```

```css
.search-input {
    background: oklch(var(--background-subtle-color));
    color: oklch(var(--foreground-color));
    border: 1px solid oklch(var(--border-color));
    border-radius: 12px;
    padding: var(--size-md);
    font-family: var(--font-fk-grotesk-neue);
    font-weight: var(--font-normal);
    letter-spacing: 0.01em;
}

.pill {
    background: oklch(var(--background-raised-color));
    border: 1px solid oklch(var(--border-color));
    border-radius: 9999px;
    padding: var(--size-sm) var(--size-md);
    transition: background 150ms ease;
}

.pill:hover {
    background: oklch(var(--background-elevated-color));
}
```

---

## Notes

- This design system uses OKLCH color space for better perceptual uniformity
- The cyan accent color is the primary brand color throughout
- Dark mode is the primary experience
- Interface prioritizes clean, spacious layout with clear hierarchy
- Monospace font available for code/technical content
