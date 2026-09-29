import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

export default function DegradationChart({ data, title, dataKeys }) {
  return (
    <div style={{ width: "100%", height: 320 }}>
      <h4>{title}</h4>
      <ResponsiveContainer>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="cycle" />
          <YAxis />
          <Tooltip />
          <Legend />
          {dataKeys.map((k) => (
            <Line key={k.key} type="monotone" dataKey={k.key} name={k.name} stroke={k.color} dot={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
