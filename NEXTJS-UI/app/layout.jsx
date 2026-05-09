export const metadata = {
  title: 'BashAgent',
  description: 'Natural language to bash commands',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, padding: 0, background: '#060606' }}>
        {children}
      </body>
    </html>
  );
}