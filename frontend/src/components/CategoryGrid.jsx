import { disposalTone } from '../utils/format.js'

/**
 * Reference list of the model's classes, from `GET /categories`.
 *
 * Each item is `{ name, index, category, icon, bin_colour, instructions }`.
 *
 * @param {{ classes: import('../services/types.js').ClassInfo[] }}
 */
export default function CategoryGrid({ classes }) {
  if (!classes.length) return null

  return (
    <section className="panel categories">
      <h3 className="panel__title">Supported categories</h3>
      <p className="panel__subtitle">The 10 classes this model was trained on.</p>

      <ul className="categories__grid">
        {classes.map((item) => (
          <li
            className={`categories__item categories__item--${disposalTone(item.category)}`}
            key={item.name}
            title={item.instructions}
          >
            <span className="categories__icon" aria-hidden="true">
              {item.icon}
            </span>
            <span className="categories__name">{item.name}</span>
            <span className="categories__category">{item.category}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}