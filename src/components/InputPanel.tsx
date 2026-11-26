import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { Mode } from "@/lib/results";
import { useStore, type Display } from "@/store";

export function InputPanel() {
  const { mode, setMode, heightCm, calibrate2D, layers, analysis, set, resetLandmarks } = useStore();
  return (
    <Card size="sm">
      <CardHeader><CardTitle>設定</CardTitle></CardHeader>
      <CardContent className="space-y-3">
        <ToggleGroup type="single" variant="outline" value={mode} onValueChange={(v) => v && setMode(v as Mode)} className="w-full">
          <ToggleGroupItem value="photo" className="flex-1">写真</ToggleGroupItem>
          <ToggleGroupItem value="illust" className="flex-1">イラスト</ToggleGroupItem>
        </ToggleGroup>
        <div className="flex items-center gap-2">
          <Label htmlFor="height" className="w-24 shrink-0">{mode === "illust" ? "設定身長" : "身長"}</Label>
          <Input id="height" type="number" inputMode="decimal" min={50} max={250} step={0.1} value={heightCm}
            onChange={(e) => { const v = Number(e.target.value); if (v > 0) set({ heightCm: v }); }} />
          <span className="text-sm text-muted-foreground">cm</span>
        </div>
        <div className="flex items-center justify-between gap-2">
          <Label htmlFor="cal2d" className="leading-snug">シルエット幅で周囲長を補正<span className="block text-xs font-normal text-muted-foreground">イラスト向け。橙の線の端をドラッグして幅を合わせる</span></Label>
          <Switch id="cal2d" checked={calibrate2D} onCheckedChange={(v) => set({ calibrate2D: v })} />
        </div>
        <div className="space-y-2 text-sm">
          {([["mesh", "3Dメッシュ", "点"], ["garment", "服の範囲", "塗り"]] as const).map(([k, label, fill]) => (
            <div key={k} className="flex items-center justify-between gap-2">
              <span>{label}</span>
              <ToggleGroup type="single" size="sm" variant="outline" value={layers[k]} onValueChange={(v) => v && set({ layers: { ...layers, [k]: v as Display } })}>
                <ToggleGroupItem value="fill">{fill}</ToggleGroupItem>
                <ToggleGroupItem value="outline">輪郭</ToggleGroupItem>
                <ToggleGroupItem value="off">なし</ToggleGroupItem>
              </ToggleGroup>
            </div>
          ))}
          <label className="flex items-center justify-between gap-2">
            周囲長の線
            <Switch size="sm" checked={layers.levels} onCheckedChange={(v) => set({ layers: { ...layers, levels: v } })} />
          </label>
        </div>
        <Button variant="outline" size="sm" className="w-full" disabled={!analysis} onClick={resetLandmarks}>
          <RotateCcw />自動推定に戻す
        </Button>
      </CardContent>
    </Card>
  );
}
