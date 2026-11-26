import { FlaskConical, Copyright, Scale, ShieldCheck, Ruler } from "lucide-react";
import { useState } from "react";
import {
  AlertDialog, AlertDialogAction, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/ui/alert-dialog";

const ITEMS = [
  { icon: FlaskConical, text: "研究・評価目的のツールです" },
  { icon: Copyright, text: "使う権利のある画像だけを入れてください" },
  { icon: ShieldCheck, text: "画像は端末内だけで処理し、外部に送信しません" },
  { icon: Ruler, text: "結果は推定値で、実寸ではありません" },
  { icon: Scale, text: "利用は自己責任でお願いします" },
];

export function TermsDialog() {
  const [open, setOpen] = useState(true);
  return (
    <AlertDialog open={open}>
      <AlertDialogContent className="sm:max-w-sm">
        <AlertDialogHeader>
          <AlertDialogTitle>ご利用の前に</AlertDialogTitle>
          <AlertDialogDescription>同意のうえでお使いください。</AlertDialogDescription>
        </AlertDialogHeader>
        <ul className="space-y-2.5 text-sm">
          {ITEMS.map(({ icon: Icon, text }) => (
            <li key={text} className="flex items-center gap-2.5">
              <Icon className="size-4 shrink-0 text-muted-foreground" />{text}
            </li>
          ))}
        </ul>
        <AlertDialogFooter className="items-center sm:justify-between">
          <a href="/licenses" target="_blank" rel="noreferrer" className="text-xs text-muted-foreground underline underline-offset-2">ライセンス</a>
          <AlertDialogAction onClick={() => setOpen(false)}>同意して使う</AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
