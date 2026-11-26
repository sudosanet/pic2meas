import { ImageUp } from "lucide-react";
import { useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { cn } from "@/lib/utils";

export function ImageDrop({ onFile, large = false }: { onFile: (f: File) => void; large?: boolean }) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "image/*": [] }, multiple: false, onDropAccepted: ([f]) => onFile(f),
  });

  useEffect(() => {
    if (!large) return;
    const onPaste = (e: ClipboardEvent) => {
      const f = [...(e.clipboardData?.files ?? [])].find((x) => x.type.startsWith("image/"));
      if (f) onFile(f);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [large, onFile]);

  return (
    <div {...getRootProps()} className={cn(
      "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed text-muted-foreground transition-colors",
      large ? "absolute inset-6 text-base" : "p-3 text-sm",
      isDragActive ? "border-primary bg-primary/5" : "border-border hover:bg-accent/40",
    )}>
      <input {...getInputProps()} />
      <ImageUp className={large ? "size-10" : "size-5"} />
      {large ? <p>画像をドロップ・クリックして選択・貼り付け（Ctrl/⌘+V）</p> : <p>別の画像を選ぶ</p>}
    </div>
  );
}
