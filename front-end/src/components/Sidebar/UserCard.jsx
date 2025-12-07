export default function UserCard({ user }) {
  if (!user) return null;

  return (
    <div className="bg-white border border-[#c7ddd2] p-4 rounded-xl">
      <h3 className="font-semibold text-[#004d2a] text-lg">{user.name}</h3>
      <p className="text-sm text-[#6c8f80]">{user.user_type}</p>
      <p className="text-xs mt-1 text-[#7b9186]">
        رقم الهوية/الإقامة: {user.national_id}
      </p>
    </div>
  );
}
