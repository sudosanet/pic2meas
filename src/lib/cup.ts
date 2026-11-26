// JIS bra size: cup = bust − underbust, AA at 7.5 cm then every 2.5 cm (±1.25); underbust band in 5 cm steps.

const CUPS = ["AA", "A", "B", "C", "D", "E", "F", "G", "H", "I"] as const;

export type BraSize = { diff: number; cup: string; band: number; label: string };

export function jisBraSize(bust: number, underbust: number): BraSize {
  const diff = bust - underbust, band = Math.round(underbust / 5) * 5, i = Math.round((diff - 7.5) / 2.5);
  if (i < 0 || i >= CUPS.length) {
    const cup = i < 0 ? "AA未満" : "I超";
    return { diff, cup, band, label: `${cup}（アンダー${band}）` };
  }
  return { diff, cup: CUPS[i], band, label: `${CUPS[i]}${band}` };
}
