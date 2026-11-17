// GridironML design tokens (Material 3 palette) — shared by all UI pages.
tailwind.config = {
  "darkMode": "class",
  "theme": {
    "extend": {
      "colors": {
        "surface-dim": "#d2d9f4", "on-secondary-fixed-variant": "#783200",
        "surface-tint": "#006c4a", "primary-container": "#00855d",
        "tertiary-fixed-dim": "#ffb95f", "tertiary-container": "#a36700",
        "on-tertiary-fixed-variant": "#653e00", "outline": "#6d7a72",
        "surface-container": "#eaedff", "surface-container-high": "#e2e7ff",
        "secondary-container": "#fd761a", "on-tertiary-container": "#fffbff",
        "on-error": "#ffffff", "on-primary-fixed": "#002114",
        "secondary": "#9d4300", "background": "#faf8ff",
        "surface-container-lowest": "#ffffff", "secondary-fixed": "#ffdbca",
        "tertiary-fixed": "#ffddb8", "tertiary": "#825100",
        "secondary-fixed-dim": "#ffb690", "surface": "#faf8ff",
        "on-surface": "#131b2e", "on-primary-fixed-variant": "#005137",
        "on-primary": "#ffffff", "on-background": "#131b2e",
        "primary-fixed": "#85f8c4", "on-primary-container": "#f5fff7",
        "on-tertiary-fixed": "#2a1700", "inverse-on-surface": "#eef0ff",
        "inverse-primary": "#68dba9", "on-error-container": "#93000a",
        "surface-container-low": "#f2f3ff", "surface-variant": "#dae2fd",
        "surface-container-highest": "#dae2fd", "primary-fixed-dim": "#68dba9",
        "on-surface-variant": "#3d4a42", "inverse-surface": "#283044",
        "primary": "#006948", "outline-variant": "#bccac0",
        "on-tertiary": "#ffffff", "error-container": "#ffdad6",
        "error": "#ba1a1a", "on-secondary-container": "#5c2400",
        "surface-bright": "#faf8ff", "on-secondary": "#ffffff",
        "on-secondary-fixed": "#341100"
      },
      "borderRadius": { "DEFAULT": "0.25rem", "lg": "0.5rem", "xl": "0.75rem", "full": "9999px" },
      "spacing": {
        "margin": "1rem", "space-xl": "2rem", "margin-md": "2rem",
        "space-lg": "1.5rem", "space-md": "1rem", "space-xs": "0.25rem",
        "margin-lg": "3rem", "gutter-md": "1.5rem", "gutter": "1rem",
        "space-sm": "0.5rem", "gutter-lg": "2rem"
      },
      "fontFamily": {
        "body-lg": ["Inter"], "label-sm": ["Inter"],
        "data-display-mobile": ["Outfit"], "label-lg": ["Inter"],
        "label-md": ["Inter"], "body-sm": ["Inter"],
        "headline-lg": ["Outfit"], "headline-sm": ["Outfit"],
        "headline-xl-mobile": ["Outfit"], "headline-lg-mobile": ["Outfit"],
        "data-display": ["Outfit"], "body-md": ["Inter"],
        "headline-xl": ["Outfit"], "headline-md": ["Outfit"]
      },
      "fontSize": {
        "body-lg": ["16px", { "lineHeight": "24px", "letterSpacing": "-0.01em", "fontWeight": "400" }],
        "label-sm": ["10px", { "lineHeight": "14px", "letterSpacing": "0.04em", "fontWeight": "700" }],
        "data-display-mobile": ["24px", { "lineHeight": "28px", "letterSpacing": "-0.01em", "fontWeight": "800" }],
        "label-lg": ["14px", { "lineHeight": "20px", "letterSpacing": "0.01em", "fontWeight": "600" }],
        "label-md": ["12px", { "lineHeight": "16px", "letterSpacing": "0.02em", "fontWeight": "600" }],
        "body-sm": ["12px", { "lineHeight": "16px", "letterSpacing": "0em", "fontWeight": "400" }],
        "headline-lg": ["28px", { "lineHeight": "36px", "letterSpacing": "-0.02em", "fontWeight": "700" }],
        "headline-sm": ["16px", { "lineHeight": "24px", "letterSpacing": "0em", "fontWeight": "600" }],
        "headline-xl-mobile": ["30px", { "lineHeight": "36px", "letterSpacing": "-0.02em", "fontWeight": "800" }],
        "headline-lg-mobile": ["24px", { "lineHeight": "32px", "letterSpacing": "-0.01em", "fontWeight": "700" }],
        "data-display": ["32px", { "lineHeight": "36px", "letterSpacing": "-0.02em", "fontWeight": "800" }],
        "body-md": ["14px", { "lineHeight": "20px", "letterSpacing": "0em", "fontWeight": "400" }],
        "headline-xl": ["40px", { "lineHeight": "48px", "letterSpacing": "-0.03em", "fontWeight": "800" }],
        "headline-md": ["20px", { "lineHeight": "28px", "letterSpacing": "-0.01em", "fontWeight": "700" }]
      }
    }
  }
};
