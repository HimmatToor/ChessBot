import ChessGame from "@/components/ChessGame";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center gap-6 px-4 py-10">
      <h1 className="text-3xl font-bold">ChessBot</h1>
      <ChessGame />
    </main>
  );
}
