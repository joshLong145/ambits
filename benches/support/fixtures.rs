#![allow(dead_code)]

use std::path::PathBuf;

use ambits::symbols::merkle::content_hash;
use ambits::symbols::{FileSymbols, ProjectTree, SymbolCategory, SymbolNode};
use ambits::tracking::{ContextLedger, ReadDepth};

// ─── Rust source fixtures ─────────────────────────────────────────────────────

pub const RUST_TINY: &str = r#"fn greet(name: &str) -> &str { name }"#;

pub const RUST_SMALL: &str = r#"
pub struct Counter {
    value: u64,
}

impl Counter {
    pub fn new() -> Self {
        Self { value: 0 }
    }

    pub fn increment(&mut self) {
        self.value += 1;
    }

    pub fn get(&self) -> u64 {
        self.value
    }
}
"#;

pub const RUST_MEDIUM: &str = r#"
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Config {
    pub name: String,
    pub values: HashMap<String, String>,
}

#[derive(Debug)]
pub enum Status {
    Active,
    Inactive,
    Pending(String),
}

pub struct Manager<T> {
    items: Vec<T>,
    config: Config,
}

impl Config {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            values: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.values.insert(key.to_string(), value.to_string());
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|s| s.as_str())
    }
}

impl<T: Clone> Manager<T> {
    pub fn new(config: Config) -> Self {
        Self { items: Vec::new(), config }
    }

    pub fn add(&mut self, item: T) {
        self.items.push(item);
    }

    pub fn count(&self) -> usize {
        self.items.len()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }
}

pub trait Describable {
    fn describe(&self) -> String;
}

impl Describable for Config {
    fn describe(&self) -> String {
        format!("Config({})", self.name)
    }
}

pub fn process_items<T: Describable>(items: &[T]) -> Vec<String> {
    items.iter().map(|i| i.describe()).collect()
}
"#;

pub const RUST_LARGE: &str = r#"
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

pub mod events {
    #[derive(Debug, Clone, PartialEq)]
    pub enum EventKind {
        Created,
        Updated { field: String },
        Deleted,
        Custom(String),
    }

    #[derive(Debug, Clone)]
    pub struct Event {
        pub id: u64,
        pub kind: EventKind,
        pub payload: Option<String>,
    }

    impl Event {
        pub fn new(id: u64, kind: EventKind) -> Self {
            Self { id, kind, payload: None }
        }

        pub fn with_payload(mut self, payload: impl Into<String>) -> Self {
            self.payload = Some(payload.into());
            self
        }
    }
}

pub trait Store<T> {
    fn insert(&mut self, key: String, value: T) -> Option<T>;
    fn get(&self, key: &str) -> Option<&T>;
    fn remove(&mut self, key: &str) -> Option<T>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
}

#[derive(Debug, Default)]
pub struct HashStore<T> {
    inner: HashMap<String, T>,
    access_log: Vec<String>,
}

impl<T: Clone> Store<T> for HashStore<T> {
    fn insert(&mut self, key: String, value: T) -> Option<T> {
        self.access_log.push(format!("insert:{key}"));
        self.inner.insert(key, value)
    }

    fn get(&self, key: &str) -> Option<&T> {
        self.inner.get(key)
    }

    fn remove(&mut self, key: &str) -> Option<T> {
        self.access_log.push(format!("remove:{key}"));
        self.inner.remove(key)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<T> HashStore<T> {
    pub fn new() -> Self {
        Self { inner: HashMap::new(), access_log: Vec::new() }
    }

    pub fn access_history(&self) -> &[String] {
        &self.access_log
    }

    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.inner.keys().map(|s| s.as_str())
    }
}

#[derive(Debug)]
pub struct Pipeline<I, O> {
    stages: Vec<Box<dyn Fn(I) -> O + Send + Sync>>,
    name: String,
}

impl<I: Clone + Send + 'static, O: Send + 'static> Pipeline<I, O> {
    pub fn new(name: impl Into<String>) -> Self {
        Self { stages: Vec::new(), name: name.into() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }
}

pub struct Registry {
    handlers: HashMap<String, Arc<Mutex<dyn Fn(&str) -> String + Send>>>,
    tags: HashSet<String>,
}

impl Registry {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
            tags: HashSet::new(),
        }
    }

    pub fn register(&mut self, name: impl Into<String>, handler: impl Fn(&str) -> String + Send + 'static) {
        self.handlers.insert(name.into(), Arc::new(Mutex::new(handler)));
    }

    pub fn tag(&mut self, tag: impl Into<String>) {
        self.tags.insert(tag.into());
    }

    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }

    pub fn handler_names(&self) -> Vec<&str> {
        self.handlers.keys().map(|s| s.as_str()).collect()
    }

    pub fn invoke(&self, name: &str, input: &str) -> Option<String> {
        self.handlers.get(name).map(|h| {
            let guard = h.lock().unwrap();
            guard(input)
        })
    }
}

pub fn deduplicate<T: Eq + std::hash::Hash + Clone>(items: &[T]) -> Vec<T> {
    let mut seen = HashSet::new();
    items.iter().filter(|i| seen.insert((*i).clone())).cloned().collect()
}

pub fn chunk<T: Clone>(items: &[T], size: usize) -> Vec<Vec<T>> {
    items.chunks(size).map(|c| c.to_vec()).collect()
}
"#;

// ─── TypeScript / JavaScript source fixtures ──────────────────────────────────
// All parsed via TypescriptParser. JS is valid TS so JS_MODULE works too.

pub const TS_TINY: &str = r#"const greet = (name: string): string => name;"#;

pub const TS_SMALL: &str = r#"
interface Greeter {
    greet(name: string): string;
    farewell(name: string): string;
}

class FormalGreeter implements Greeter {
    private prefix: string;

    constructor(prefix: string) {
        this.prefix = prefix;
    }

    greet(name: string): string {
        return `${this.prefix} ${name}`;
    }

    farewell(name: string): string {
        return `Goodbye, ${name}`;
    }
}
"#;

pub const TS_MEDIUM: &str = r#"
export interface Repository<T> {
    findById(id: string): Promise<T | null>;
    findAll(): Promise<T[]>;
    save(entity: T): Promise<T>;
    delete(id: string): Promise<void>;
}

export type EntityId = string;

export interface Entity {
    id: EntityId;
    createdAt: Date;
    updatedAt: Date;
}

export enum Status {
    Active = "active",
    Inactive = "inactive",
    Pending = "pending",
}

export class User implements Entity {
    id: EntityId;
    createdAt: Date;
    updatedAt: Date;
    name: string;
    email: string;
    status: Status;

    constructor(id: EntityId, name: string, email: string) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.status = Status.Pending;
        this.createdAt = new Date();
        this.updatedAt = new Date();
    }

    activate(): void {
        this.status = Status.Active;
        this.updatedAt = new Date();
    }

    deactivate(): void {
        this.status = Status.Inactive;
        this.updatedAt = new Date();
    }
}

export namespace Events {
    export interface UserCreated {
        userId: EntityId;
        timestamp: Date;
    }

    export interface UserDeleted {
        userId: EntityId;
        timestamp: Date;
        reason?: string;
    }
}

export class EventBus<T extends object> {
    private handlers: Map<string, Array<(event: T) => void>> = new Map();

    subscribe(eventType: string, handler: (event: T) => void): void {
        const existing = this.handlers.get(eventType) ?? [];
        this.handlers.set(eventType, [...existing, handler]);
    }

    publish(eventType: string, event: T): void {
        const handlers = this.handlers.get(eventType) ?? [];
        handlers.forEach(h => h(event));
    }

    unsubscribeAll(eventType: string): void {
        this.handlers.delete(eventType);
    }
}
"#;

pub const TS_LARGE: &str = r#"
import { EventEmitter } from 'events';

export type DeepPartial<T> = T extends object
    ? { [K in keyof T]?: DeepPartial<T[K]> }
    : T;

export type Result<T, E = Error> =
    | { ok: true; value: T }
    | { ok: false; error: E };

export function ok<T>(value: T): Result<T> {
    return { ok: true, value };
}

export function err<E extends Error>(error: E): Result<never, E> {
    return { ok: false, error };
}

export interface Logger {
    debug(msg: string, ...args: unknown[]): void;
    info(msg: string, ...args: unknown[]): void;
    warn(msg: string, ...args: unknown[]): void;
    error(msg: string, ...args: unknown[]): void;
}

export class ConsoleLogger implements Logger {
    private readonly prefix: string;

    constructor(prefix: string) {
        this.prefix = prefix;
    }

    debug(msg: string, ...args: unknown[]): void {
        console.debug(`[${this.prefix}] ${msg}`, ...args);
    }

    info(msg: string, ...args: unknown[]): void {
        console.info(`[${this.prefix}] ${msg}`, ...args);
    }

    warn(msg: string, ...args: unknown[]): void {
        console.warn(`[${this.prefix}] ${msg}`, ...args);
    }

    error(msg: string, ...args: unknown[]): void {
        console.error(`[${this.prefix}] ${msg}`, ...args);
    }
}

export abstract class BaseService {
    protected readonly logger: Logger;
    protected readonly name: string;

    constructor(name: string, logger: Logger) {
        this.name = name;
        this.logger = logger;
    }

    abstract initialize(): Promise<void>;
    abstract shutdown(): Promise<void>;

    protected log(level: 'debug' | 'info' | 'warn' | 'error', msg: string): void {
        this.logger[level](`[${this.name}] ${msg}`);
    }
}

export enum CacheStrategy {
    LRU = 'lru',
    LFU = 'lfu',
    FIFO = 'fifo',
}

export class Cache<K, V> {
    private readonly store: Map<K, { value: V; hits: number; insertedAt: number }>;
    private readonly maxSize: number;
    private readonly strategy: CacheStrategy;

    constructor(maxSize: number, strategy: CacheStrategy = CacheStrategy.LRU) {
        this.store = new Map();
        this.maxSize = maxSize;
        this.strategy = strategy;
    }

    get(key: K): V | undefined {
        const entry = this.store.get(key);
        if (!entry) return undefined;
        entry.hits++;
        return entry.value;
    }

    set(key: K, value: V): void {
        if (this.store.size >= this.maxSize) {
            this.evict();
        }
        this.store.set(key, { value, hits: 0, insertedAt: Date.now() });
    }

    has(key: K): boolean {
        return this.store.has(key);
    }

    delete(key: K): boolean {
        return this.store.delete(key);
    }

    clear(): void {
        this.store.clear();
    }

    get size(): number {
        return this.store.size;
    }

    private evict(): void {
        const entries = [...this.store.entries()];
        let victim: K;
        switch (this.strategy) {
            case CacheStrategy.LFU:
                victim = entries.sort(([, a], [, b]) => a.hits - b.hits)[0][0];
                break;
            case CacheStrategy.FIFO:
                victim = entries.sort(([, a], [, b]) => a.insertedAt - b.insertedAt)[0][0];
                break;
            default:
                victim = entries[0][0];
        }
        this.store.delete(victim);
    }
}

export class HttpClient {
    private readonly baseUrl: string;
    private readonly headers: Record<string, string>;
    private readonly timeout: number;

    constructor(baseUrl: string, options: { headers?: Record<string, string>; timeout?: number } = {}) {
        this.baseUrl = baseUrl;
        this.headers = options.headers ?? {};
        this.timeout = options.timeout ?? 5000;
    }

    async get<T>(path: string): Promise<Result<T>> {
        try {
            const response = await fetch(`${this.baseUrl}${path}`, {
                headers: this.headers,
                signal: AbortSignal.timeout(this.timeout),
            });
            if (!response.ok) {
                return err(new Error(`HTTP ${response.status}`));
            }
            const data = await response.json() as T;
            return ok(data);
        } catch (e) {
            return err(e instanceof Error ? e : new Error(String(e)));
        }
    }

    async post<T>(path: string, body: unknown): Promise<Result<T>> {
        try {
            const response = await fetch(`${this.baseUrl}${path}`, {
                method: 'POST',
                headers: { ...this.headers, 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
                signal: AbortSignal.timeout(this.timeout),
            });
            if (!response.ok) {
                return err(new Error(`HTTP ${response.status}`));
            }
            const data = await response.json() as T;
            return ok(data);
        } catch (e) {
            return err(e instanceof Error ? e : new Error(String(e)));
        }
    }
}

export class EventQueue extends EventEmitter {
    private readonly queue: unknown[] = [];
    private processing = false;

    enqueue(event: unknown): void {
        this.queue.push(event);
        if (!this.processing) {
            this.process();
        }
    }

    private async process(): Promise<void> {
        this.processing = true;
        while (this.queue.length > 0) {
            const event = this.queue.shift();
            this.emit('event', event);
        }
        this.processing = false;
    }

    get pending(): number {
        return this.queue.length;
    }
}
"#;

pub const JS_MODULE: &str = r#"
class EventBus {
    constructor() {
        this.handlers = new Map();
    }

    subscribe(eventType, handler) {
        const existing = this.handlers.get(eventType) ?? [];
        this.handlers.set(eventType, [...existing, handler]);
    }

    publish(eventType, event) {
        const handlers = this.handlers.get(eventType) ?? [];
        handlers.forEach(h => h(event));
    }
}

class UserService {
    constructor(db, bus) {
        this.db = db;
        this.bus = bus;
    }

    async createUser(name, email) {
        const user = await this.db.insert({ name, email });
        this.bus.publish('user.created', { userId: user.id });
        return user;
    }

    async deleteUser(id) {
        await this.db.delete(id);
        this.bus.publish('user.deleted', { userId: id });
    }

    async getUser(id) {
        return this.db.findById(id);
    }
}

const formatUser = user => `${user.name} <${user.email}>`;

const validateEmail = email => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);

const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);

const memoize = fn => {
    const cache = new Map();
    return (...args) => {
        const key = JSON.stringify(args);
        if (cache.has(key)) return cache.get(key);
        const result = fn(...args);
        cache.set(key, result);
        return result;
    };
};

function retry(fn, maxAttempts = 3, delayMs = 100) {
    return async function(...args) {
        let lastError;
        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            try {
                return await fn(...args);
            } catch (e) {
                lastError = e;
                if (attempt < maxAttempts - 1) {
                    await new Promise(r => setTimeout(r, delayMs * (attempt + 1)));
                }
            }
        }
        throw lastError;
    };
}

module.exports = { EventBus, UserService, formatUser, validateEmail, pipe, memoize, retry };
"#;

// ─── Python source fixtures ───────────────────────────────────────────────────

pub const PY_TINY: &str = "def greet(name):\n    return name\n";

pub const PY_SMALL: &str = r#"
import functools

def log_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


class Counter:
    def __init__(self, start: int = 0) -> None:
        self._value = start

    def increment(self) -> None:
        self._value += 1

    def decrement(self) -> None:
        self._value -= 1

    @property
    def value(self) -> int:
        return self._value

    def reset(self) -> None:
        self._value = 0


@log_call
def process(items: list) -> list:
    return [x for x in items if x is not None]


@log_call
def summarize(data: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in data.items())
"#;

pub const PY_MEDIUM: &str = r#"
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Iterator
import asyncio

MAX_RETRIES: int = 3
DEFAULT_TIMEOUT: float = 30.0
EMPTY_RESULT: list = []


@dataclass
class Config:
    host: str
    port: int = 8080
    timeout: float = DEFAULT_TIMEOUT
    retries: int = MAX_RETRIES
    tags: list[str] = field(default_factory=list)

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def with_tag(self, tag: str) -> Config:
        return Config(
            host=self.host,
            port=self.port,
            timeout=self.timeout,
            retries=self.retries,
            tags=[*self.tags, tag],
        )


@dataclass
class Response:
    status: int
    body: str
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> dict:
        import json
        return json.loads(self.body)


@dataclass
class RequestStats:
    total: int = 0
    success: int = 0
    failure: int = 0

    def record(self, ok: bool) -> None:
        self.total += 1
        if ok:
            self.success += 1
        else:
            self.failure += 1

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.success / self.total


type ConfigOrNone = Optional[Config]


async def fetch(url: str, timeout: float = DEFAULT_TIMEOUT) -> Response:
    await asyncio.sleep(0)
    return Response(status=200, body="{}")


async def fetch_with_retry(url: str, config: Config) -> Response:
    last_exc: Optional[Exception] = None
    for attempt in range(config.retries):
        try:
            return await fetch(url, config.timeout)
        except Exception as exc:
            last_exc = exc
            await asyncio.sleep(0.1 * (attempt + 1))
    raise RuntimeError(f"Failed after {config.retries} attempts") from last_exc


def paginate(items: list, page_size: int) -> Iterator[list]:
    for i in range(0, len(items), page_size):
        yield items[i : i + page_size]
"#;

pub const PY_LARGE: &str = r#"
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Generic, Iterator, Optional, TypeVar
import asyncio
import functools
import logging

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

logger = logging.getLogger(__name__)

MAX_POOL_SIZE: int = 100
DEFAULT_BATCH_SIZE: int = 50
IDLE_TIMEOUT_SECS: float = 60.0


def retry(max_attempts: int = 3, delay: float = 0.1):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    logger.warning(f"Attempt {attempt + 1} failed: {exc}")
                    await asyncio.sleep(delay * (attempt + 1))
            raise RuntimeError(f"All {max_attempts} attempts failed") from last_exc
        return wrapper
    return decorator


def log_access(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        logger.debug(f"{type(self).__name__}.{func.__name__} called")
        return func(self, *args, **kwargs)
    return wrapper


class Repository(ABC, Generic[T, K]):
    @abstractmethod
    async def find_by_id(self, id: K) -> Optional[T]:
        ...

    @abstractmethod
    async def find_all(self) -> list[T]:
        ...

    @abstractmethod
    async def save(self, entity: T) -> T:
        ...

    @abstractmethod
    async def delete(self, id: K) -> None:
        ...


@dataclass
class CacheEntry(Generic[V]):
    value: V
    hits: int = 0
    inserted_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())

    def touch(self) -> None:
        self.hits += 1


class LRUCache(Generic[K, V]):
    def __init__(self, max_size: int) -> None:
        self._store: dict[K, CacheEntry[V]] = {}
        self._max_size = max_size

    def get(self, key: K) -> Optional[V]:
        entry = self._store.get(key)
        if entry is None:
            return None
        entry.touch()
        return entry.value

    def set(self, key: K, value: V) -> None:
        if len(self._store) >= self._max_size:
            self._evict()
        self._store[key] = CacheEntry(value=value)

    def invalidate(self, key: K) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def _evict(self) -> None:
        if not self._store:
            return
        victim = min(self._store, key=lambda k: self._store[k].inserted_at)
        del self._store[victim]

    @property
    def size(self) -> int:
        return len(self._store)


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}
        self._history: list[tuple[str, Any]] = []

    def subscribe(self, event_type: str, handler) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler) -> None:
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event_type: str, payload: Any) -> None:
        self._history.append((event_type, payload))
        for handler in self._handlers.get(event_type, []):
            if asyncio.iscoroutinefunction(handler):
                await handler(payload)
            else:
                handler(payload)

    def history(self) -> Iterator[tuple[str, Any]]:
        yield from self._history


@dataclass
class PipelineStage:
    name: str
    transform: Any
    enabled: bool = True


class Pipeline(Generic[T]):
    def __init__(self, name: str) -> None:
        self.name = name
        self._stages: list[PipelineStage] = []

    def add_stage(self, name: str, transform, enabled: bool = True) -> Pipeline[T]:
        self._stages.append(PipelineStage(name=name, transform=transform, enabled=enabled))
        return self

    async def run(self, input_data: T) -> T:
        result = input_data
        for stage in self._stages:
            if not stage.enabled:
                continue
            if asyncio.iscoroutinefunction(stage.transform):
                result = await stage.transform(result)
            else:
                result = stage.transform(result)
        return result

    def stage_names(self) -> list[str]:
        return [s.name for s in self._stages if s.enabled]


def batch_items(items: list[T], size: int) -> Iterator[list[T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def gather_results(coros) -> list:
    return await asyncio.gather(*coros, return_exceptions=True)


async def stream_process(items: list[T], processor) -> AsyncIterator[Any]:
    for item in items:
        if asyncio.iscoroutinefunction(processor):
            result = await processor(item)
        else:
            result = processor(item)
        yield result
"#;

// ─── Runtime fixture builders ─────────────────────────────────────────────────

/// Create a minimal SymbolNode for use in benchmarks.
pub fn make_sym(id: &str, name: &str) -> SymbolNode {
    let hash = content_hash(name);
    SymbolNode {
        id: id.to_string(),
        name: name.to_string(),
        category: SymbolCategory::Function,
        label: "fn".to_string(),
        file_path: PathBuf::from("src/bench.rs"),
        byte_range: 0..100,
        line_range: 1..10,
        content_hash: hash,
        merkle_hash: hash,
        children: Vec::new(),
        estimated_tokens: 30,
    }
}

/// Build a recursive symbol tree: each node has `breadth` children, `depth` levels deep.
pub fn make_symbol_tree(depth: usize, breadth: usize) -> SymbolNode {
    fn build(depth: usize, breadth: usize, prefix: &str) -> SymbolNode {
        let id = format!("{prefix}::node");
        let mut node = make_sym(&id, &id);
        if depth > 0 {
            node.children = (0..breadth)
                .map(|i| build(depth - 1, breadth, &format!("{prefix}::{i}")))
                .collect();
        }
        ambits::symbols::merkle::compute_merkle_hash(&mut node);
        node
    }
    build(depth, breadth, "root")
}

/// Build a flat list of N symbols, all in one file.
pub fn make_flat_symbols(n: usize) -> Vec<SymbolNode> {
    (0..n).map(|i| make_sym(&format!("src/bench.rs::sym_{i}"), &format!("sym_{i}"))).collect()
}

/// Build a FileSymbols containing `n_symbols` flat symbols.
pub fn make_file(path: &str, n_symbols: usize) -> FileSymbols {
    let file_path = PathBuf::from(path);
    let symbols = make_flat_symbols(n_symbols)
        .into_iter()
        .map(|mut s| { s.file_path = file_path.clone(); s })
        .collect();
    FileSymbols { file_path, symbols, total_lines: n_symbols * 5 }
}

/// Build a ProjectTree with `n_files` files, each containing `symbols_per_file` symbols.
pub fn make_project(n_files: usize, symbols_per_file: usize) -> ProjectTree {
    let files = (0..n_files)
        .map(|i| make_file(&format!("src/module_{i}.rs"), symbols_per_file))
        .collect();
    ProjectTree { root: PathBuf::from("/bench/project"), files }
}

/// Build a ContextLedger pre-populated with entries for all symbols in `project`.
pub fn make_populated_ledger(project: &ProjectTree) -> ContextLedger {
    let mut ledger = ContextLedger::new();
    let hash = [0u8; 32];
    for file in &project.files {
        for sym in &file.symbols {
            ledger.record(sym.id.clone(), ReadDepth::FullBody, hash, "agent-0".to_string(), sym.estimated_tokens);
        }
    }
    ledger
}

/// Generate N realistic assistant JSONL lines, cycling through common tool types.
pub fn make_jsonl_lines(n: usize) -> String {
    let tools = [
        (r#"{"type":"read","file_path":"src/main.rs"}"#, "Read"),
        (r#"{"pattern":"**/*.rs"}"#, "Glob"),
        (r#"{"pattern":"fn parse","path":"src"}"#, "Grep"),
        (r#"{"name_path_pattern":"parse_file","relative_path":"src/parser/rust.rs"}"#, "mcp__serena__find_symbol"),
        (r#"{"relative_path":"src/coverage.rs","depth":1}"#, "mcp__serena__get_symbols_overview"),
    ];
    let mut out = String::with_capacity(n * 200);
    for i in 0..n {
        let (input, tool) = tools[i % tools.len()];
        let session = format!("session-{}", i / 100);
        out.push_str(&format!(
            r#"{{"type":"assistant","sessionId":"{session}","agentId":"agent-{i}","timestamp":"2025-01-01T00:00:00Z","message":{{"role":"assistant","content":[{{"type":"tool_use","name":"{tool}","input":{input}}}]}}}}"#
        ));
        out.push('\n');
    }
    out
}

/// Return the source `const` for a given size tag (used by parser benchmarks).
pub fn rust_source(size: &str) -> &'static str {
    match size {
        "tiny" => RUST_TINY,
        "small" => RUST_SMALL,
        "medium" => RUST_MEDIUM,
        "large" => RUST_LARGE,
        _ => RUST_TINY,
    }
}

pub fn ts_source(size: &str) -> &'static str {
    match size {
        "tiny" => TS_TINY,
        "small" => TS_SMALL,
        "medium" => TS_MEDIUM,
        "large" => TS_LARGE,
        "js_module" => JS_MODULE,
        _ => TS_TINY,
    }
}

pub fn py_source(size: &str) -> &'static str {
    match size {
        "tiny" => PY_TINY,
        "small" => PY_SMALL,
        "medium" => PY_MEDIUM,
        "large" => PY_LARGE,
        _ => PY_TINY,
    }
}

/// Build a realistic single-tool JSONL line for ingest benchmarks.
pub fn jsonl_read_line() -> &'static str {
    r#"{"type":"assistant","sessionId":"s1","agentId":"a1","timestamp":"2025-01-01T00:00:00Z","message":{"role":"assistant","content":[{"type":"tool_use","name":"Read","input":{"file_path":"src/coverage.rs"}}]}}"#
}

pub fn jsonl_multi_tool_line() -> String {
    // Five tool_use blocks in one assistant message
    let tools: Vec<&str> = vec![
        r#"{"type":"tool_use","name":"Read","input":{"file_path":"src/main.rs"}}"#,
        r#"{"type":"tool_use","name":"Glob","input":{"pattern":"**/*.rs"}}"#,
        r#"{"type":"tool_use","name":"Grep","input":{"pattern":"fn parse","path":"src"}}"#,
        r#"{"type":"tool_use","name":"mcp__serena__find_symbol","input":{"name_path_pattern":"parse_file"}}"#,
        r#"{"type":"tool_use","name":"mcp__serena__get_symbols_overview","input":{"relative_path":"src/lib.rs","depth":1}}"#,
    ];
    let content = tools.join(",");
    format!(
        r#"{{"type":"assistant","sessionId":"s1","agentId":"a1","timestamp":"2025-01-01T00:00:00Z","message":{{"role":"assistant","content":[{content}]}}}}"#
    )
}

pub fn jsonl_malformed_line() -> &'static str {
    r#"{"type":"assistant","broken":true,"missing_message_key":"#
}
