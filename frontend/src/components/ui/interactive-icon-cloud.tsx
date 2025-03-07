"use client"

import { useEffect, useMemo, useState } from "react"
import { useTheme } from "next-themes"
import {
  Cloud,
  fetchSimpleIcons,
  ICloud,
  renderSimpleIcon,
  SimpleIcon,
} from "react-icon-cloud"

// Update version in the URL when fetching icons
const SIMPLE_ICONS_VERSION = "14.0.0" // Match the version being used by the system

export const cloudProps: Omit<ICloud, "children"> = {
  containerProps: {
    style: {
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      width: "100%",
      height: "100%",
      padding: "20px 0",
    },
  },
  options: {
    reverse: true,
    depth: 0.8,
    wheelZoom: false,
    imageScale: 1.75,
    activeCursor: "pointer",
    tooltip: "native",
    clickToFront: 500,
    tooltipDelay: 0,
    outlineColour: "#0000",
    maxSpeed: 0.04,
    minSpeed: 0.02,
    radiusX: 0.9,
    radiusY: 0.9,
    radiusZ: 0.9,
    stretchX: 1,
    stretchY: 1,
    offsetX: 0,
    offsetY: 0,
    shuffleTags: true,
    initial: [0.1, -0.1],
    fadeIn: 500,
    zoom: 0.95,
    dragControl: true,
    noSelect: false,
    noMouse: false,
    pinchZoom: true,
    freezeActive: true,
    freezeDecel: true,
    shape: "sphere"
  },
}

// Map company tickers to their SimpleIcon slugs
const tickerToSlug: Record<string, string> = {
  // Tech Companies
  AAPL: "apple",
  GOOGL: "google",
  MSFT: "microsoft",
  AMZN: "amazonaws",
  META: "facebook",
  NFLX: "netflix",
  TSLA: "tesla",
  NVDA: "nvidia",
  ADBE: "adobe",
  ORCL: "oracle",
  CSCO: "cisco",
  IBM: "ibm",
  INTC: "intel",
  AMD: "amd",
  QCOM: "qualcomm",
  CRM: "salesforce",
  AVGO: "broadcom",
  HPQ: "hp",
  DELL: "dell",
  VMW: "vmware",
  SAP: "sap",
  INTU: "intuit",
  WDAY: "workday",
  NOW: "servicenow",
  PLTR: "palantir",
  AMAT: "applied",
  RHT: "redhat",
  PANW: "paloalto",
  SPLK: "splunk",
  ZEN: "zendesk",
  CRWD: "crowdstrike",
  
  // Consumer & Social Media
  DIS: "disney",
  SBUX: "starbucks",
  ABNB: "airbnb",
  PG: "procter-and-gamble",
  KO: "coca-cola",
  PEP: "pepsi",
  MCD: "mcdonalds",
  WMT: "walmart",
  TGT: "target",
  COST: "costco",
  NKE: "nike",
  LEVI: "levis",
  UAA: "underarmour",
  CROX: "crocs",
  ROST: "ross",
  VFC: "vans",
  PUMA: "puma",
  ADS: "adidas",
  
  // Financial & Payment
  JPM: "jpmorgan",
  V: "visa",
  MA: "mastercard",
  PYPL: "paypal",
  MS: "morgan-stanley",
  GS: "goldman-sachs",
  BAC: "bank-of-america",
  C: "citigroup",
  WFC: "wells-fargo",
  AXP: "american-express",
  SQ: "square",
  COIN: "coinbase",
  HOOD: "robinhood",
  SCHW: "charles-schwab",
  BLK: "blackrock",
  SOFI: "sofi",
  WU: "westernunion",
  AFRM: "affirm",
  STRIPE: "stripe",
  KLARNA: "klarna",
  
  // E-commerce & Delivery
  SHOP: "shopify",
  ETSY: "etsy",
  EBAY: "ebay",
  DASH: "doordash",
  UBER: "uber",
  LYFT: "lyft",
  BABA: "aliexpress",
  MELI: "mercadolibre",
  JD: "jd",
  INSTACART: "instacart",
  GRUB: "grubhub",
  
  // Cloud & Enterprise
  TEAM: "atlassian",
  ZS: "zscaler",
  NET: "cloudflare",
  DDOG: "datadog",
  MDB: "mongodb",
  SNOW: "snowflake",
  DBX: "dropbox",
  BOX: "box",
  ZM: "zoom",
  OKTA: "okta",
  TWLO: "twilio",
  FSLY: "fastly",
  DOCN: "digitalocean",
  RXT: "rackspace",
  GTLB: "gitlab",
  HUBS: "hubspot",
  SUMO: "sumologic",
  NEWR: "newrelic",
  ELASTIC: "elasticsearch",
  NGINX: "nginx",
  REDIS: "redis",
  PUPPET: "puppet",
  ANSIBLE: "ansible",
  DOCKER: "docker",
  K8S: "kubernetes",
  
  // Developer Tools & Platforms
  GH: "github",
  GITLAB: "gitlab",
  BITBUCKET: "bitbucket",
  NPM: "npm",
  YARN: "yarn",
  BABEL: "babel",
  WEBPACK: "webpack",
  VITE: "vite",
  VERCEL: "vercel",
  NETLIFY: "netlify",
  HEROKU: "heroku",
  JENKINS: "jenkins",
  TRAVIS: "travis-ci",
  CIRCLE: "circleci",
  JIRA: "jira",
  CONFLUENCE: "confluence",
  
  // Social & Communication
  SNAP: "snapchat",
  PINS: "pinterest",
  MTCH: "tinder",
  BMBL: "bumble",
  DISCORD: "discord",
  SLACK: "slack",
  TEAMS: "microsoft-teams",
  TELEGRAM: "telegram",
  SIGNAL: "signal",
  LINE: "line",
  WECHAT: "wechat",
  
  // Gaming & Entertainment
  NTDOY: "nintendo",
  SONY: "playstation",
  EA: "ea",
  TTWO: "rockstar-games",
  ATVI: "battle-net",
  UNITY: "unity",
  RBLX: "roblox",
  EPIC: "epic-games",
  STEAM: "steam",
  TWITCH: "twitch",
  
  // Software & Development
  ADSK: "autodesk",
  ANSS: "ansys",
  AZPN: "aspen",
  CTXS: "citrix",
  FTNT: "fortinet",
  NLOK: "norton",
  AVAST: "avast",
  MCAFEE: "mcafee",
  SYMANTEC: "symantec",
  
  // Browsers & Web
  CHROME: "google-chrome",
  FIREFOX: "firefox",
  SAFARI: "safari",
  EDGE: "edge",
  BRAVE: "brave",
  OPERA: "opera",
  TOR: "tor-project",
  
  // Others
  PTON: "peloton",
  SWX: "swiss-exchange",
  WISE: "wise",
  DOCS: "notion",
  FVRR: "fiverr",
  UPWK: "upwork",
  ASANA: "asana",
  MONDAY: "monday",
  TRELLO: "trello",
  FIGMA: "figma",
  CANVA: "canva",
  ADOBE: "adobe",
  SKETCH: "sketch",
  INVISION: "invision",
  DRIBBBLE: "dribbble",
  BEHANCE: "behance",
  
  // Cryptocurrency
  "BTC-USD": "bitcoin",
  "ETH-USD": "ethereum",
  "SOL-USD": "solana",
  "ADA-USD": "cardano",
  "DOT-USD": "polkadot",
  "AVAX-USD": "avalancheavax",
  "MATIC-USD": "polygon",
  "LINK-USD": "chainlink",
  "XRP-USD": "xrp",
  "DOGE-USD": "dogecoin",
  "SHIB-USD": "shibainu",
  "UNI-USD": "uniswap",
  "AAVE-USD": "aave",
  "ATOM-USD": "cosmos",
  "ALGO-USD": "algorand",
  "FTM-USD": "fantom",
  "NEAR-USD": "near",
  "BNB-USD": "binance",
  "LTC-USD": "litecoin",
  "XLM-USD": "stellar",
  "TRX-USD": "tron",
  "EOS-USD": "eos",
  "ZEC-USD": "zcash",
  "DASH-USD": "dashpay",
  "XMR-USD": "monero",
  "NEO-USD": "neo",
  "CAKE-USD": "pancakeswap",
  "SUSHI-USD": "sushiswap",
  "GRT-USD": "graphql",
  "COMP-USD": "compound",
  "MKR-USD": "maker",
  "YFI-USD": "yearn",
  "SNX-USD": "synthetix",
  "BAT-USD": "brave",
  "1INCH-USD": "1inch",
  "THETA-USD": "theta",
  "VET-USD": "vechain",
  "WAVES-USD": "waves",
  "IOTA-USD": "iota",
  "XTZ-USD": "tezos",
  "FLOW-USD": "flow",
  "MANA-USD": "decentraland",
  "SAND-USD": "thesandboxgame",
  "AXS-USD": "axieinfinity",
  "ENJ-USD": "enjin",
  "CHZ-USD": "chiliz",
  "HOT-USD": "holochain",
  "ONE-USD": "harmony",
  "EGLD-USD": "elrond",
  "HBAR-USD": "hedera",
  "ICX-USD": "icon",
  "ZIL-USD": "zilliqa",
  "ONT-USD": "ontology",
  "QTUM-USD": "qtum",
  "DGB-USD": "digibyte",
  "RVN-USD": "ravencoin",
  "STORJ-USD": "storj",
  "FIL-USD": "filecoin",
  "AR-USD": "arweave",
  "ANKR-USD": "ankr",
  "BAND-USD": "bandprotocol",
  "KAVA-USD": "kava",
  "LUNA-USD": "terra",
  "CRV-USD": "curve",
  "REN-USD": "renproject",
  "BAL-USD": "balancer",
  "PERP-USD": "perpetual",
  "KNC-USD": "kyber",
  "OCEAN-USD": "ocean-protocol",
  "API3-USD": "api3",
  "ALPHA-USD": "alpha",
  "ORN-USD": "orion",
  "RUNE-USD": "thorchain",
}

export const renderCustomIcon = (icon: SimpleIcon, theme: string, onClick?: () => void) => {
  const bgHex = theme === "light" ? "#f3f2ef" : "#080510"
  const fallbackHex = theme === "light" ? "#6e6e73" : "#ffffff"
  const minContrastRatio = theme === "dark" ? 2 : 1.2

  return renderSimpleIcon({
    icon,
    bgHex,
    fallbackHex,
    minContrastRatio,
    size: 36,
    aProps: {
      href: undefined,
      target: undefined,
      rel: undefined,
      onClick: (e: any) => {
        e.preventDefault()
        onClick?.()
      },
      style: {
        cursor: onClick ? 'pointer' : 'default',
      }
    },
  })
}

export type DynamicCloudProps = {
  tickers: string[]
  onTickerSelect: (ticker: string) => void
}

type IconData = Awaited<ReturnType<typeof fetchSimpleIcons>>

export function IconCloud({ tickers, onTickerSelect }: DynamicCloudProps) {
  const [data, setData] = useState<IconData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const { theme } = useTheme()

  useEffect(() => {
    const slugs = tickers
      .filter(ticker => {
        // Check if the ticker exists in our mapping
        const hasMapping = tickerToSlug[ticker];
        if (!hasMapping) {
          console.debug(`No icon mapping found for ticker: ${ticker}`);
        }
        return hasMapping;
      })
      .map(ticker => {
        const slug = tickerToSlug[ticker];
        console.debug(`Mapping ${ticker} to icon: ${slug}`);
        return slug;
      });
    
    if (slugs.length === 0) {
      console.debug('No valid icons to display');
      return;
    }

    console.debug('Fetching icons for slugs:', slugs);
    
    fetchSimpleIcons({ 
      slugs
    })
    .then(data => {
      console.debug('Successfully fetched icons:', Object.keys(data.simpleIcons));
      setData(data);
    })
    .catch(err => {
      console.error('Error fetching icons:', err);
      setError('Failed to load icons');
    });
  }, [tickers]);

  const renderedIcons = useMemo(() => {
    if (!data) return null

    return Object.values(data.simpleIcons).map((icon) => {
      // Find the ticker for this icon
      const ticker = Object.entries(tickerToSlug).find(([_, slug]) => slug === icon.slug)?.[0]
      if (!ticker) return null

      return renderCustomIcon(
        icon,
        theme || "light",
        () => onTickerSelect(ticker)
      )
    }).filter(Boolean)
  }, [data, theme, onTickerSelect])

  return (
    <div className="w-full h-[500px] relative bg-black/20 rounded-lg overflow-hidden">
      {/* @ts-ignore */}
      <Cloud {...cloudProps}>
        <>{renderedIcons}</>
      </Cloud>
    </div>
  )
} 