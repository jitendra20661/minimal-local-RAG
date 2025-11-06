# LAQ RAG Web App Design Guide

## Table of Contents
1. [Design Philosophy](#design-philosophy)
2. [Visual Identity](#visual-identity)
3. [Color System](#color-system)
4. [Typography](#typography)
5. [Layout & Grid](#layout--grid)
6. [Component Library](#component-library)
7. [Interaction Patterns](#interaction-patterns)
8. [Responsive Design](#responsive-design)
9. [Accessibility](#accessibility)
10. [Animation & Motion](#animation--motion)

---

## Design Philosophy

### Core Principles

**1. Minimal & Focused**
- Remove visual noise and distractions
- One primary action per screen
- Progressive disclosure of information
- White space is a design element, not empty space

**2. Privacy-First Interface**
- No external tracking indicators
- Local-only operations highlighted
- Clear data flow visualization
- Transparent processing states

**3. Information Clarity**
- Clear hierarchy: Primary → Secondary → Tertiary
- Scannable layouts with visual anchors
- Context-aware information density
- Search-first mentality

**4. Professional & Technical**
- Clean, modern aesthetic
- Technical sophistication without complexity
- Data visualization where appropriate
- Developer-friendly design patterns

**5. Speed & Efficiency**
- Keyboard shortcuts for power users
- Quick actions readily available
- Minimal clicks to complete tasks
- Instant visual feedback

---

## Visual Identity

### Brand Characteristics

**Personality Traits:**
- **Intelligent** - Smart RAG technology
- **Trustworthy** - Privacy-focused, local processing
- **Efficient** - Fast, streamlined workflows
- **Modern** - Contemporary design language
- **Technical** - Developer/researcher oriented

**Visual Mood:**
- Dark, sophisticated interface
- Calm, focused environment
- High-tech but approachable
- Clean and professional

### Logo/Brand Mark

**Primary Mark:**
- Geometric abstract shape suggesting:
  - Network/connections (RAG retrieval)
  - Database structure (vector storage)
  - Question mark (Q&A pairs)
- Central dot/circle in accent color representing the "answer"
- Minimal, scalable design

**Usage:**
- Always maintain clear space (minimum 8px on all sides)
- Never rotate or distort
- Single color in UI contexts
- Can be animated subtly (pulse on activity)

---

## Color System

### Color Philosophy
Use OKLCH color space for perceptual uniformity and better dark mode rendering. All colors should be semantically named and used consistently.

### Dark Theme Palette (Primary)

#### Background Layers
```css
/* Deepest to Lightest */
--background-base: #100e12        /* Main canvas, deepest layer */
--background-subtle: #1a1820      /* Slightly raised surfaces */
--background-subtler: #252229     /* Cards, panels */
--background-raised: #2d2a33      /* Elevated elements, hover states */
--background-elevated: #38343f    /* Modals, popovers, highest elevation */
```

**Usage Rules:**
- Base: Page background, main canvas
- Subtle: First-level containers (sidebar, main sections)
- Subtler: Cards, list items, input backgrounds
- Raised: Hover states, active elements
- Elevated: Modals, dropdowns, overlays

#### Foreground/Text Colors
```css
/* Most to Least Prominent */
--foreground-color: #e8e6e3           /* Primary text, headings */
--foreground-quiet: #c4c2be           /* Secondary text, labels */
--foreground-quieter: #a09d98         /* Tertiary text, captions */
--foreground-subtle: #7c7975          /* Placeholder text, disabled */
--foreground-subtler: #5a5854         /* Dividers, subtle borders */
```

**Usage Rules:**
- Primary text: Main content, headings, important labels
- Quiet: Descriptions, secondary information
- Quieter: Metadata, timestamps, helper text
- Subtle: Placeholders, disabled text
- Subtler: Borders, dividers (avoid for text)

#### Accent Colors

**Primary Accent - Cyan/Teal**
```css
--accent-primary: #00d4c4             /* Main brand color */
--accent-primary-hover: #00f5e3       /* Hover, active states */
--accent-primary-subtle: rgba(0, 212, 196, 0.15)  /* Backgrounds */
--accent-primary-border: rgba(0, 212, 196, 0.3)   /* Borders */
```

**When to Use:**
- Primary CTAs and important actions
- Active navigation states
- Focus indicators
- Links and interactive elements
- Success indicators (alternative to green)
- Loading states and progress

**Semantic Colors**
```css
--positive: #4ade80       /* Success, confirmations, additions */
--negative: #f87171       /* Errors, deletions, warnings (critical) */
--caution: #fbbf24        /* Warnings, alerts (non-critical) */
--attention: #60a5fa      /* Information, notices, tips */
```

**Usage Guidelines:**
- Positive: Successful uploads, query matches, data added
- Negative: Failed operations, validation errors, deletion confirmations
- Caution: Low relevance scores, missing data warnings
- Attention: Helper tips, informational messages, feature highlights

#### Border & Shadow Colors
```css
--border-default: rgba(255, 255, 255, 0.1)
--border-hover: rgba(255, 255, 255, 0.15)
--border-focus: var(--accent-primary)
--shadow-overlay: rgba(0, 0, 0, 0.7)
```

### Color Usage Rules

**DO:**
- ✓ Use semantic color variables, never hardcode
- ✓ Maintain minimum 4.5:1 contrast for text
- ✓ Use accent color sparingly for emphasis
- ✓ Test colors in context (against actual backgrounds)
- ✓ Layer backgrounds progressively for depth

**DON'T:**
- ✗ Use pure white (#ffffff) or pure black (#000000)
- ✗ Mix semantic colors (don't use green for errors)
- ✗ Use more than 2 accent colors simultaneously
- ✗ Create custom colors outside the system
- ✗ Use color as the only differentiator (accessibility)

### Color Combinations

**Safe Pairings:**
```
Background Subtle + Foreground Primary (high contrast)
Background Raised + Foreground Quiet (medium contrast)
Accent Primary + Background Base (maximum impact)
Background Elevated + Border Default (subtle separation)
```

**Avoid:**
```
Foreground Subtle + Background Subtle (too low contrast)
Multiple semantic colors together (visual confusion)
Accent on accent (overwhelming)
```

---

## Typography

### Font System

**Primary Font Stack:**
```css
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
             Roboto, Helvetica, Arial, sans-serif;
```

**Fallback Order:**
1. System fonts (optimal performance)
2. If custom fonts loaded: FK Grotesk Neue or similar geometric sans
3. Never load heavy web fonts (maintain speed)

### Font Weights (Dark Mode Optimized)

```css
--font-light: 275
--font-normal: 375        /* Body text default */
--font-medium: 475        /* Emphasized text, labels */
--font-semibold: 575      /* Subheadings, important labels */
--font-bold: 675          /* Headings, key metrics */
--font-extrabold: 775     /* Large display text (rare) */
```

**Why Lighter Weights?**
Dark backgrounds make text appear heavier due to light halation. Reduce standard weights by 25-100 units for better optical balance.

### Type Scale

```css
--font-xs: 11px           /* Fine print, metadata */
--font-sm: 13px           /* Secondary text, captions */
--font-base: 15px         /* Body text, default */
--font-md: 16px           /* Emphasized body text */
--font-lg: 18px           /* Section headings */
--font-xl: 20px           /* Page headings */
--font-2xl: 24px          /* Large headings */
--font-3xl: 32px          /* Display text, stats */
--font-4xl: 48px          /* Hero text (rare) */
```

### Line Heights

```css
--leading-tight: 1.25     /* Headings, large text */
--leading-snug: 1.375     /* Subheadings */
--leading-normal: 1.5     /* Body text (default) */
--leading-relaxed: 1.625  /* Long-form content */
--leading-loose: 1.75     /* Captions, fine print */
```

### Letter Spacing

```css
--tracking-tight: -0.01em     /* Large headings */
--tracking-normal: 0.005em    /* Light mode body */
--tracking-wide: 0.01em       /* Dark mode body (default) */
--tracking-wider: 0.02em      /* Uppercase, small caps */
```

**Dark Mode Adjustment:**
Increase letter spacing by 0.005-0.01em for better legibility on dark backgrounds.

### Typography Hierarchy

**Level 1: Page Title**
```css
font-size: 20px
font-weight: 575
line-height: 1.25
letter-spacing: -0.01em
color: var(--foreground-color)
```
*Use for: Main page headings, dashboard title*

**Level 2: Section Heading**
```css
font-size: 18px
font-weight: 575
line-height: 1.375
letter-spacing: 0
color: var(--foreground-color)
```
*Use for: Section titles, modal headers*

**Level 3: Subsection Heading**
```css
font-size: 16px
font-weight: 475
line-height: 1.5
letter-spacing: 0.005em
color: var(--foreground-color)
```
*Use for: Card titles, list headings*

**Body Text**
```css
font-size: 15px
font-weight: 375
line-height: 1.5
letter-spacing: 0.01em
color: var(--foreground-color)
```
*Use for: Primary content, descriptions*

**Secondary Text**
```css
font-size: 14px
font-weight: 375
line-height: 1.5
letter-spacing: 0.01em
color: var(--foreground-quiet)
```
*Use for: Labels, metadata, helper text*

**Caption Text**
```css
font-size: 13px
font-weight: 375
line-height: 1.5
letter-spacing: 0.01em
color: var(--foreground-quieter)
```
*Use for: Timestamps, fine print, supplementary info*

**Display Numbers (Stats)**
```css
font-size: 32px
font-weight: 575
line-height: 1.25
letter-spacing: -0.01em
color: var(--foreground-color)
font-variant-numeric: tabular-nums
```
*Use for: Large statistics, metrics*

### Typography Best Practices

**DO:**
- ✓ Use tabular numbers for data tables and stats
- ✓ Limit line length to 70-80 characters for readability
- ✓ Use hierarchy to guide the eye (size, weight, color)
- ✓ Maintain consistent vertical rhythm (multiples of 4px)
- ✓ Test readability at actual size (not zoomed)

**DON'T:**
- ✗ Use more than 3 font sizes on a single screen
- ✗ Center-align long text blocks
- ✗ Use all caps for long text (titles only)
- ✗ Mix multiple font families
- ✗ Use font weights below 275 on dark backgrounds

---

## Layout & Grid

### Layout Principles

**1. Central Focus**
- Primary content centered on screen
- Maximum width constraints for readability
- Generous whitespace on larger screens

**2. Fixed Navigation**
- Sidebar always visible (desktop)
- Bottom nav on mobile
- Consistent positioning builds muscle memory

**3. Progressive Complexity**
- Simple actions prominent and easy
- Advanced features discoverable but not intrusive
- Drill-down for details (don't show everything at once)

### Grid System

**Spacing Scale (8px Base Unit)**
```css
--space-2xs: 2px      /* Tight spacing, icon gaps */
--space-xs: 4px       /* Very tight spacing */
--space-sm: 8px       /* Compact spacing, button padding */
--space-md: 16px      /* Standard spacing (default) */
--space-ml: 24px      /* Medium-large spacing */
--space-lg: 32px      /* Large spacing, section gaps */
--space-xl: 48px      /* Extra large, major sections */
--space-2xl: 64px     /* Huge gaps (rare) */
```

**Usage:**
- All spacing should be multiples of 4px or 8px
- Use md (16px) as default for most spacing
- Section gaps: lg (32px) or xl (48px)
- Component internal padding: sm (8px) to md (16px)

### Layout Dimensions

**Sidebar:**
```css
--sidebar-width: 72px              /* Desktop collapsed */
--sidebar-width-expanded: 220px    /* Desktop expanded */
```

**Content Widths:**
```css
--content-max-width: 1100px        /* Main content container */
--content-reading-width: 740px     /* Optimal reading width */
--content-narrow-width: 600px      /* Forms, focused content */
```

**Header/Navigation:**
```css
--header-height: 54px              /* Top header */
--mobile-nav-height: 72px          /* Bottom mobile nav */
```

**Touch Targets:**
```css
--min-touch-target: 44px           /* Minimum clickable area */
--preferred-touch-target: 48px     /* Preferred clickable area */
```

### Common Layout Patterns

#### Dashboard Layout
```
┌─────┬──────────────────────────┐
│     │ Header (54px)            │
│ S   ├──────────────────────────┤
│ I   │                          │
│ D   │ Content (centered)       │
│ E   │ max-width: 1100px        │
│ B   │                          │
│ A   │                          │
│ R   │                          │
│     │                          │
│ 72px│                          │
└─────┴──────────────────────────┘
```

#### Search-Focused Layout
```
┌─────┬──────────────────────────┐
│     │ Header                   │
│ S   ├──────────────────────────┤
│ I   │         [Search]         │
│ D   │       max: 740px         │
│ E   │      [Suggestions]       │
│ B   ├──────────────────────────┤
│ A   │                          │
│ R   │ Results (centered)       │
│     │                          │
└─────┴──────────────────────────┘
```

#### Modal/Dialog
```
Centered, max-width: 600px
Padding: 24px
Backdrop: rgba(0,0,0,0.7) + blur
Border radius: 16px
```

### Responsive Breakpoints

```css
--breakpoint-sm: 640px      /* Small phones */
--breakpoint-md: 768px      /* Tablets, large phones */
--breakpoint-lg: 1024px     /* Laptops, small desktops */
--breakpoint-xl: 1280px     /* Desktops */
--breakpoint-2xl: 1536px    /* Large desktops */
```

**Breakpoint Strategy:**
- Mobile first: Design for 375px-640px first
- Tablet: 768px (sidebar collapses to bottom nav)
- Desktop: 1024px+ (full sidebar visible)

---

## Component Library

### Buttons

#### Primary Button
**Use for:** Main actions, CTAs, form submissions

**Specs:**
```css
background: var(--accent-primary)
color: var(--background-base)
padding: 8px 16px
border-radius: 8px
font-size: 14px
font-weight: 475
min-height: 44px
border: none
transition: all 150ms ease
```

**States:**
- Default: Solid accent color
- Hover: Lighter accent (`--accent-primary-hover`)
- Active: Slightly darker, scale(0.98)
- Disabled: 40% opacity, no pointer events
- Focus: 2px outline with accent color, 2px offset

#### Secondary Button
**Use for:** Alternative actions, cancel operations

**Specs:**
```css
background: var(--background-subtle)
color: var(--foreground-color)
border: 1px solid var(--border-default)
/* Other specs same as primary */
```

**States:**
- Hover: `background-subtler`, border brighter

#### Ghost Button
**Use for:** Tertiary actions, less important options

**Specs:**
```css
background: transparent
color: var(--foreground-quiet)
border: none
/* Padding and sizing same */
```

**States:**
- Hover: `background-subtle`

#### Icon Button
**Use for:** Toolbar actions, compact interfaces

**Specs:**
```css
width: 36px
height: 36px
padding: 0
border-radius: 6px
background: transparent
color: var(--accent-primary)
```

### Input Fields

#### Text Input
**Specs:**
```css
background: var(--background-subtle)
border: 1px solid var(--border-default)
border-radius: 8px (small) / 12px (large)
padding: 8px 16px (small) / 16px 24px (large)
color: var(--foreground-color)
font-size: 14px (small) / 15px (large)
min-height: 44px
```

**States:**
- Default: Subtle background, light border
- Focus: Accent border, slightly brighter background
- Error: Red border, error message below
- Disabled: 50% opacity, no pointer events

**Placeholder:**
```css
color: var(--foreground-subtle)
font-weight: 375
```

#### Search Input
**Special variant with:**
- Larger size (16px padding, min-height 48px)
- Rounded corners (12px)
- Icon toolbar inside input
- Accent-colored icons

### Cards

#### Standard Card
**Use for:** Content containers, list items, info panels

**Specs:**
```css
background: var(--background-subtle)
border: 1px solid var(--border-default)
border-radius: 12px
padding: 16px (compact) / 24px (standard)
transition: all 150ms ease
```

**States:**
- Default: Subtle background
- Hover: Raised background, brighter border
- Active/Selected: Accent border or left accent bar

#### Stat Card
**Use for:** Metrics, key numbers

**Structure:**
- Label (small, quiet color)
- Value (large, bold)
- Change indicator (colored, with icon)

**Specs:**
```css
/* Same base as standard card */
min-width: 200px
```

#### Activity Card
**Use for:** Lists, timelines, activity feeds

**Structure:**
- Icon (colored, in circle or square)
- Title + metadata
- Timestamp (right-aligned)

### Modals & Overlays

#### Modal
**Specs:**
```css
max-width: 600px (standard) / 900px (large)
border-radius: 16px
background: var(--background-elevated)
border: 1px solid var(--border-default)
```

**Structure:**
- Header: Title + close button (54px height)
- Body: Main content (24px padding)
- Footer (optional): Actions right-aligned

**Backdrop:**
```css
background: rgba(0, 0, 0, 0.7)
backdrop-filter: blur(4px)
```

#### Tooltip
**Specs:**
```css
background: var(--background-elevated)
border: 1px solid var(--border-default)
border-radius: 6px
padding: 6px 12px
font-size: 13px
max-width: 200px
box-shadow: 0 4px 12px rgba(0,0,0,0.3)
```

### Navigation

#### Sidebar Navigation
**Item Specs:**
```css
width: 100%
padding: 8px
border-radius: 8px
display: flex
flex-direction: column
align-items: center
gap: 4px
```

**States:**
- Default: Transparent, subtle text
- Hover: Subtle background
- Active: Accent color text + accent background

**Icon Size:** 24px
**Label Size:** 11px

#### Pills/Tags
**Use for:** Suggestions, filters, categories

**Specs:**
```css
padding: 8px 16px
border-radius: 9999px (fully rounded)
background: var(--background-raised)
border: 1px solid var(--border-default)
font-size: 14px
font-weight: 475
display: inline-flex
align-items: center
gap: 8px
```

### Lists

#### Activity List
**Item Structure:**
- Icon (40px circle/square, colored background)
- Content (flex-grow)
  - Title (15px, medium weight)
  - Metadata (13px, quiet color)
- Time (13px, subtle color, right-aligned)

**Separators:**
- 1px border between items (`border-color`)
- No border on last item

### Progress Indicators

#### Loading Spinner
**Specs:**
```css
size: 20px (small) / 32px (medium) / 48px (large)
color: var(--accent-primary)
stroke-width: 2px
animation: spin 1s linear infinite
```

#### Progress Bar
**Specs:**
```css
height: 4px (thin) / 8px (thick)
background: var(--background-raised)
border-radius: 9999px
```

**Fill:**
```css
background: linear-gradient(90deg,
  var(--accent-primary),
  var(--accent-primary-hover))
border-radius: 9999px
transition: width 300ms ease
```

### Alerts/Notifications

#### Toast Notification
**Specs:**
```css
min-width: 300px
max-width: 400px
background: var(--background-elevated)
border: 1px solid var(--border-default)
border-radius: 12px
padding: 16px
box-shadow: 0 8px 24px rgba(0,0,0,0.4)
```

**Variants:**
- Success: Left accent bar in green
- Error: Left accent bar in red
- Warning: Left accent bar in yellow
- Info: Left accent bar in blue

**Position:** Top-right, 24px margin

---

## Interaction Patterns

### Microinteractions

#### Hover States
**Timing:** 150ms ease transition
**Changes:**
- Background: Lighten by one step
- Border: Increase opacity by 5%
- Scale: Generally avoid (except buttons can scale(0.98) on active)

#### Click/Tap Feedback
**Active State:**
```css
transform: scale(0.98)
transition: transform 100ms ease
```

**Ripple Effect (optional):**
- Subtle expanding circle from click point
- Accent color at 10% opacity
- 400ms duration, ease-out

#### Focus States
**All interactive elements:**
```css
outline: 2px solid var(--accent-primary)
outline-offset: 2px
border-radius: inherit
```

**Never remove focus indicators** - critical for accessibility

### Page Transitions

**Page Load:**
- Fade in content (200ms)
- Stagger list items (50ms delay each)

**Route Changes:**
- Fade out current (150ms)
- Fade in new (150ms) with 50ms delay
- Total transition: 350ms

**Modal Open/Close:**
- Backdrop: Fade in 200ms
- Content: Scale from 0.95 to 1, fade in 200ms
- Close: Reverse, 150ms

### Loading States

**Inline Loading:**
- Show spinner next to text
- Disable controls but keep visible
- Show "Loading..." text for screen readers

**Full-Screen Loading:**
- Centered spinner
- Optional loading message
- Semi-transparent overlay

**Skeleton Screens:**
- Use for lists and cards
- Animate with subtle shimmer
- Match final content layout

### Drag & Drop

**Upload Zone:**
- Default: Dashed border
- Hover: Solid border, accent color
- Drag over: Filled accent background (15% opacity)
- Drop: Brief success animation

**States:**
```css
/* Default */
border: 2px dashed var(--border-default)

/* Drag Over */
border: 2px solid var(--accent-primary)
background: var(--accent-primary-subtle)
```

### Empty States

**Components:**
- Icon (48px, subtle color)
- Title (16px, medium weight)
- Description (14px, quiet color)
- Optional CTA button

**Tone:** Friendly, encouraging, actionable

**Examples:**
- "No PDFs uploaded yet" → "Upload your first LAQ PDF"
- "No search results" → "Try different keywords"

### Error States

**Inline Errors (Forms):**
- Red border on input
- Error icon (16px) next to input
- Error message below input (13px, red)

**Page-Level Errors:**
- Error icon (48px)
- Error title (18px)
- Error description with suggested action
- Primary CTA to resolve

**Toast Errors:**
- Red left accent bar
- Error icon
- Brief message + optional action button

---

## Responsive Design

### Mobile-First Approach

**Start with:** 375px width (iPhone SE)
**Scale up to:** 2560px+ (large desktops)

### Responsive Patterns

#### Sidebar Navigation
**Desktop (>768px):**
- Fixed left sidebar, 72px width
- Always visible
- Icon + label vertical layout

**Mobile (<768px):**
- Fixed bottom navigation
- Full width
- Icon + label horizontal layout
- 4-5 items maximum

#### Content Layout
**Desktop:**
- Centered content, max-width 1100px
- Side margins auto
- Multi-column grids where appropriate

**Mobile:**
- Full width with 16px horizontal padding
- Single column
- Stack all grid items

#### Modals
**Desktop:**
- Centered on screen
- Max-width 600px
- Backdrop with blur

**Mobile:**
- Full screen or bottom sheet
- No side margins (or minimal 8px)
- Can slide up from bottom

#### Tables/Data
**Desktop:**
- Full table with all columns

**Mobile:**
- Card-based layout
- Most important data visible
- "Show more" for additional columns
- Or horizontal scroll with sticky first column

### Breakpoint-Specific Rules

**< 640px (Mobile):**
- Font sizes: Reduce by 1-2px
- Padding: Reduce to 12px-16px
- Touch targets: Minimum 44px height
- Single column layouts

**640px - 1024px (Tablet):**
- 2-column grids where appropriate
- Sidebar can be toggleable
- Comfortable padding (16px-24px)

**> 1024px (Desktop):**
- Full layouts
- Multi-column grids (2-4 columns)
- Hover states enabled
- Maximum content width constraints

---

## Accessibility

### WCAG 2.1 AA Compliance

#### Color Contrast
**Minimum Ratios:**
- Normal text (< 18px): 4.5:1
- Large text (≥ 18px): 3:1
- UI components: 3:1

**Testing:**
- Use contrast checker tools
- Test all text/background combinations
- Never rely on color alone

#### Keyboard Navigation
**Requirements:**
- All interactive elements focusable
- Logical tab order (top to bottom, left to right)
- Focus indicators always visible
- Keyboard shortcuts documented

**Key Patterns:**
- Tab: Next element
- Shift + Tab: Previous element
- Enter/Space: Activate
- Escape: Close modals/dropdowns
- Arrow keys: Navigate lists/menus

#### Screen Reader Support
**ARIA Labels:**
- All icon buttons need aria-label
- Complex widgets need proper ARIA roles
- Form inputs need associated labels
- Error messages use aria-live

**Semantic HTML:**
- Use `<button>` not `<div onclick>`
- Use `<nav>` for navigation
- Use `<main>` for main content
- Use heading hierarchy (h1 → h2 → h3)

#### Focus Management
**Modal Open:**
- Move focus to modal
- Trap focus inside modal
- Return focus on close

**Dynamic Content:**
- Announce changes to screen readers
- Use aria-live regions
- Provide loading state announcements

### Touch & Mobile Accessibility

**Touch Targets:**
- Minimum 44px × 44px
- Preferred 48px × 48px
- Adequate spacing (8px minimum between targets)

**Gestures:**
- Never require complex gestures
- Provide alternatives to swipe/pinch
- Support single-finger interactions

### Reduced Motion

**Respect user preferences:**
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## Animation & Motion

### Animation Principles

**Purpose-Driven:**
- Guide attention
- Provide feedback
- Show relationships
- Indicate state changes

**Subtle & Fast:**
- Most animations: 150-300ms
- Never block user interaction
- Can be disabled for accessibility

### Timing Functions

```css
--ease-out: cubic-bezier(0.33, 1, 0.68, 1)      /* Decelerating */
--ease-in: cubic-bezier(0.32, 0, 0.67, 0)       /* Accelerating */
--ease-in-out: cubic-bezier(0.65, 0, 0.35, 1)   /* Smooth both ends */
--ease-bounce: cubic-bezier(0.68, -0.55, 0.265, 1.55)  /* Playful */
```

**When to Use:**
- Ease-out: Entrances, appearing elements (most common)
- Ease-in: Exits, disappearing elements
- Ease-in-out: Continuous movement
- Bounce: Playful interactions (use sparingly)

### Common Animations

#### Fade In/Out
```css
/* Fade In */
opacity: 0 → 1
duration: 200ms
timing: ease-out

/* Fade Out */
opacity: 1 → 0
duration: 150ms
timing: ease-in
```

#### Slide In
```css
/* From Bottom */
transform: translateY(20px) → translateY(0)
opacity: 0 → 1
duration: 250ms
timing: ease-out
```

#### Scale
```css
/* Grow */
transform: scale(0.95) → scale(1)
duration: 200ms
timing: ease-out

/* Shrink (feedback) */
transform: scale(1) → scale(0.98)
duration: 100ms
timing: ease-in-out
```

#### Loading Spinner
```css
/* Continuous rotation */
transform: rotate(0deg) → rotate(360deg)
duration: 1000ms
timing: linear
iteration: infinite
```

#### Skeleton Shimmer
```css
/* Background position animation */
background: linear-gradient(90deg,
  transparent 0%,
  rgba(255,255,255,0.05) 50%,
  transparent 100%)
background-size: 200% 100%
animation: shimmer 1.5s ease-in-out infinite

@keyframes shimmer {
  0% { background-position: -200% 0 }
  100% { background-position: 200% 0 }
}
```

### Animation Duration Guidelines

**Micro-interactions:** 100-150ms
- Button clicks
- Hover states
- Toggle switches

**Small Animations:** 150-250ms
- Dropdowns
- Tooltips
- Small modals

**Medium Animations:** 250-350ms
- Page transitions
- Large modals
- Drawer slides

**Large Animations:** 350-500ms
- Full-page transitions
- Complex choreography

**Never exceed:** 500ms (feels sluggish)

### Stagger Effects

**List Items:**
```css
/* Each item delayed by 50ms */
item:nth-child(1) { animation-delay: 0ms }
item:nth-child(2) { animation-delay: 50ms }
item:nth-child(3) { animation-delay: 100ms }
/* etc., max 10 items */
```

**Usage:**
- Reveal lists sequentially
- Draw attention to new content
- Create flow and rhythm

### Performance Considerations

**Use GPU-accelerated properties only:**
- ✓ transform (translate, scale, rotate)
- ✓ opacity
- ✗ width, height, top, left, margin (slow)

**Optimize:**
```css
/* Add to animated elements */
will-change: transform, opacity;

/* Remove after animation */
animation-end: remove will-change
```

**Never animate:**
- More than 20 elements simultaneously
- During critical user interactions
- On slow devices (use media query)

---

## Design Workflow

### Design → Development Handoff

**Deliverables:**
1. **Component Specs**
   - Exact dimensions and spacing
   - Color variables (not hex codes)
   - Typography settings
   - State variations

2. **Interaction Notes**
   - Hover, focus, active states
   - Animation timings and easing
   - Conditional logic

3. **Responsive Behavior**
   - Breakpoint changes
   - Mobile adaptations
   - Touch target adjustments

4. **Accessibility Requirements**
   - ARIA labels needed
   - Keyboard interactions
   - Focus order

### Design System Maintenance

**Regular Reviews:**
- Quarterly: Review and update color system
- As needed: Add new components to library
- Ongoing: Document exceptions and one-offs

**Version Control:**
- Document all changes to design system
- Maintain changelog
- Communicate updates to team

---

## Quick Reference Checklist

### Before Designing a New Screen

- [ ] Review existing patterns in component library
- [ ] Identify primary user goal (one per screen)
- [ ] Sketch information hierarchy
- [ ] Consider mobile layout first
- [ ] Plan keyboard navigation flow

### Before Handing Off to Development

- [ ] All colors use semantic variables
- [ ] All spacing uses spacing scale
- [ ] All touch targets minimum 44px
- [ ] Focus states designed
- [ ] Error states designed
- [ ] Empty states designed
- [ ] Loading states designed
- [ ] Mobile responsive design included
- [ ] Accessibility annotations added
- [ ] Animation specs documented

### Design Review Checklist

- [ ] Follows brand visual identity
- [ ] Maintains design system consistency
- [ ] Achieves WCAG AA contrast ratios
- [ ] Has clear visual hierarchy
- [ ] Supports keyboard navigation
- [ ] Works on mobile (375px width)
- [ ] Has appropriate feedback for all interactions
- [ ] Handles edge cases (errors, empty, loading)
- [ ] Animations are subtle and fast (<300ms)
- [ ] Text is legible and scannable

---

## Resources & Tools

### Design Tools
- **Figma/Sketch:** Component library
- **Contrast Checker:** WCAG compliance
- **Responsively:** Responsive testing
- **OKLCH Color Picker:** Color selection

### Developer Tools
- **CSS Variables:** Semantic color system
- **Autoprefixer:** Browser compatibility
- **Lighthouse:** Accessibility audit
- **axe DevTools:** A11y testing

### Inspiration
- Perplexity AI: Dark UI patterns
- Linear: Clean, minimal interface
- Raycast: Command-driven UX
- Vercel: Developer-focused design

---

## Conclusion

This design guide is a living document. As the LAQ RAG application evolves, so should this guide. The goal is to maintain **consistency**, ensure **accessibility**, and create a **delightful user experience** that makes complex RAG technology feel simple and approachable.

**Core Takeaways:**
1. **Dark mode first** - Optimize for focused, distraction-free work
2. **Semantic colors** - Never hardcode, always use variables
3. **Accessibility is non-negotiable** - Build for everyone
4. **Speed matters** - Fast animations, instant feedback
5. **Progressive disclosure** - Simple by default, powerful when needed

When in doubt, refer to this guide. When the guide doesn't cover it, extend it thoughtfully and document your decisions.
