function SourceList({ sources }) {

  if (!sources || sources.length === 0) return null;

  return (
    <div style={{ marginTop: "10px", fontSize: "14px", color: "#aaa" }}>

      <strong>Sources:</strong>

      <ul>
        {sources.map((src, index) => (
          <li key={index}>{src}</li>
        ))}
      </ul>

    </div>
  );
}

export default SourceList;