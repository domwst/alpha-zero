import { render } from 'preact';

import { App } from './App';
import { Experiments } from './Experiments';
import './styles.css';

if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
  document.documentElement.classList.add('motion-on');
}

render(location.pathname === '/experiments' ? <Experiments /> : <App />, document.getElementById('app')!);
