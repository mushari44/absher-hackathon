export default function PageContainer({ children }) {
  return (
    <div className="flex flex-row w-full min-h-screen bg-[#f5f7f6]">
      {children}
    </div>
  );
}
