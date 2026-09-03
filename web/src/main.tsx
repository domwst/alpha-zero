import { render } from 'preact';

import { App } from './App';
import './styles.css';

if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
  document.documentElement.classList.add('motion-on');
}

render(<App />, document.getElementById('app')!);
