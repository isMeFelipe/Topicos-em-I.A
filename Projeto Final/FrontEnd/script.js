document.getElementById('match-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const id1 = document.getElementById('id1').value;
  const id2 = document.getElementById('id2').value;

  const res = await fetch('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id1, id2 })
  });

  const data = await res.json();
  const div = document.getElementById('resultado');
  if (data.error) {
    div.innerHTML = `<p style="color:red">${data.error}</p>`;
  } else {
    div.innerHTML = `
      <p><strong>Resultado:</strong> ${data.resultado}</p>
      <p>Confiança: ${data.probabilidade}%</p>
    `;
  }
});
