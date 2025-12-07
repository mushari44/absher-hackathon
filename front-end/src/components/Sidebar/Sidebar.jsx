import UserCard from "./UserCard";
import UserSelector from "./UserSelector";

export default function Sidebar({ users, currentUser, currentUserKey, onSwitch }) {
  return (
    <aside className="w-72 bg-[#eaf4f0] border-l border-[#c7ddd2] p-5 flex flex-col gap-6">
      <div className="flex items-center gap-3">
        <span className="text-3xl">🤖</span>
        <div>
          <h2 className="font-bold text-[#004d2a] text-lg">أبشر مساعد</h2>
          <p className="text-xs text-[#4d6b5c]">النموذج التجريبي</p>
        </div>
      </div>

      <UserCard user={currentUser} />

      <UserSelector 
        users={users}
        currentUserKey={currentUserKey}
        onSwitch={onSwitch}
      />
    </aside>
  );
}
