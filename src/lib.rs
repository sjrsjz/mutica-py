use std::sync::OnceLock;

use mutica::{
    mutica_compiler::{
        self, ariadne,
        parser::{
            BuildContext, MultiFileBuilder, MultiFileBuilderError, ParseContext, PatternCounter,
            SyntaxError, ast::LinearizeContext,
        },
    },
    mutica_core::{
        arc_gc::{arc::GCArcWeak, gc::GC, traceable::GCTraceable},
        scheduler::{self, LinearScheduler},
        types::{
            AsDispatcher, CoinductiveType, GcAllocObject, TaggedPtr, Type, TypeError,
            character::Character, character_value::CharacterValue, float::Float,
            float_value::FloatValue, generalize::Generalize, integer::Integer,
            integer_value::IntegerValue, lazy::Lazy, list::List, namespace::Namespace,
            opcode::Opcode, rot::Rotate, specialize::Specialize, tuple::Tuple,
            type_bound::TypeBound,
        },
        util::{cycle_detector::FastCycleDetector, rootstack::RootStack},
    },
};
use pyo3::{
    Bound, IntoPyObjectExt, PyAny, PyResult, Python, pymodule,
    types::{
        PyAnyMethods, PyBool, PyDict, PyFloat, PyInt, PyList, PyModule, PyModuleMethods, PyNone,
        PyString, PyTuple,
    },
};

// 定义一个用于GC堆分配的类型
pub struct TypeGcOnceLock {
    inner: OnceLock<Type<TypeGcOnceLock>>,
}

impl GcAllocObject<TypeGcOnceLock> for TypeGcOnceLock {
    /// 创建未初始化的不动点占位符
    ///
    /// 这是递归类型定义的第一步：创建一个"洞"，稍后填充.
    type Inner = Type<TypeGcOnceLock>;

    fn new_placeholder() -> Self {
        TypeGcOnceLock {
            inner: OnceLock::new(),
        }
    }

    fn get_inner(&self) -> Option<&Self::Inner> {
        self.inner.get()
    }

    fn map_inner<F, R>(
        &self,
        path: &mut FastCycleDetector<TaggedPtr<()>>,
        f: F,
    ) -> Result<R, TypeError<Self::Inner, TypeGcOnceLock>>
    where
        F: FnOnce(
            &mut FastCycleDetector<TaggedPtr<()>>,
            <Self::Inner as AsDispatcher<Self::Inner, TypeGcOnceLock>>::RefDispatcher<'_>,
        ) -> R,
    {
        match self.inner.get() {
            Some(t) => path
                .with_guard(t.tagged_ptr(), |path| t.map_inner(path, f))
                .ok_or(TypeError::InfiniteRecursion)?,
            None => Err(TypeError::UnresolvableType),
        }
    }

    fn set_inner(&self, _value: Self::Inner) -> Result<(), TypeError<Self::Inner, TypeGcOnceLock>> {
        self.inner
            .set(_value)
            .map_err(|_| TypeError::RedeclaredType)
    }
}

impl GCTraceable<TypeGcOnceLock> for TypeGcOnceLock {
    fn collect(&self, queue: &mut std::collections::VecDeque<GCArcWeak<TypeGcOnceLock>>) {
        if let Some(t) = self.inner.get() {
            t.collect(queue);
        }
    }
}

#[pyo3::pyclass]
pub struct MuticaType {
    ty: Type<TypeGcOnceLock>,
}

#[pyo3::pymethods]
impl MuticaType {
    #[staticmethod]
    pub fn integer() -> Self {
        MuticaType::from_type(Integer::new())
    }
    #[staticmethod]
    pub fn char() -> Self {
        MuticaType::from_type(Character::new())
    }
    #[staticmethod]
    pub fn float() -> Self {
        MuticaType::from_type(Float::new())
    }
    #[staticmethod]
    pub fn integer_value(value: i64) -> Self {
        MuticaType::from_type(IntegerValue::new(value))
    }
    #[staticmethod]
    pub fn char_value(value: char) -> Self {
        MuticaType::from_type(CharacterValue::new(value))
    }
    #[staticmethod]
    pub fn float_value(value: f64) -> Self {
        MuticaType::from_type(FloatValue::new(value))
    }
    #[staticmethod]
    pub fn namespace(tag: &str, value: &MuticaType) -> Self {
        MuticaType::from_type(Namespace::new(tag, value.ty.clone()))
    }
    #[staticmethod]
    pub fn tuple(py: pyo3::Python<'_>, elements: Vec<pyo3::Py<MuticaType>>) -> Self {
        let element_types = elements
            .into_iter()
            .map(|e| e.borrow(py).ty.clone())
            .collect::<Vec<_>>();
        MuticaType::from_type(Tuple::new(element_types))
    }
    #[staticmethod]
    pub fn top() -> Self {
        MuticaType::from_type(TypeBound::top())
    }
    #[staticmethod]
    pub fn bottom() -> Self {
        MuticaType::from_type(TypeBound::bottom())
    }
    #[staticmethod]
    pub fn list(py: pyo3::Python<'_>, elements: Vec<pyo3::Py<MuticaType>>) -> Self {
        let element_types = elements
            .into_iter()
            .map(|e| e.borrow(py).ty.clone())
            .collect::<Vec<_>>();
        MuticaType::from_type(List::new(element_types))
    }
    #[staticmethod]
    pub fn opcode_add() -> Self {
        MuticaType::from_type(Opcode::Add.dispatch())
    }
    #[staticmethod]
    pub fn opcode_sub() -> Self {
        MuticaType::from_type(Opcode::Sub.dispatch())
    }
    #[staticmethod]
    pub fn opcode_mul() -> Self {
        MuticaType::from_type(Opcode::Mul.dispatch())
    }
    #[staticmethod]
    pub fn opcode_div() -> Self {
        MuticaType::from_type(Opcode::Div.dispatch())
    }
    #[staticmethod]
    pub fn opcode_mod() -> Self {
        MuticaType::from_type(Opcode::Mod.dispatch())
    }
    #[staticmethod]
    pub fn opcode_less() -> Self {
        MuticaType::from_type(Opcode::Less.dispatch())
    }
    #[staticmethod]
    pub fn opcode_greater() -> Self {
        MuticaType::from_type(Opcode::Greater.dispatch())
    }
    #[staticmethod]
    pub fn opcode_is() -> Self {
        MuticaType::from_type(Opcode::Is.dispatch())
    }
    #[staticmethod]
    pub fn opcode_neg() -> Self {
        MuticaType::from_type(Opcode::Neg.dispatch())
    }
    #[staticmethod]
    pub fn opcode_io(io_name: &str) -> Self {
        MuticaType::from_type(Opcode::IO(io_name.to_string().into()).dispatch())
    }
    #[staticmethod]
    pub fn opcode() -> Self {
        MuticaType::from_type(Opcode::Opcode.dispatch())
    }
    #[staticmethod]
    pub fn generalize(py: pyo3::Python<'_>, elements: Vec<pyo3::Py<MuticaType>>) -> Self {
        let element_types = elements
            .into_iter()
            .map(|e| e.borrow(py).ty.clone())
            .collect::<Vec<_>>();
        MuticaType::from_type(Generalize::new_raw(element_types))
    }
    #[staticmethod]
    pub fn specialize(py: pyo3::Python<'_>, elements: Vec<pyo3::Py<MuticaType>>) -> Self {
        let element_types = elements
            .into_iter()
            .map(|e| e.borrow(py).ty.clone())
            .collect::<Vec<_>>();
        MuticaType::from_type(Specialize::new_raw(element_types))
    }
    #[staticmethod]
    pub fn rot(value: &MuticaType) -> Self {
        MuticaType::from_type(Rotate::new(value.ty.clone()))
    }
    #[staticmethod]
    pub fn lazy(value: &MuticaType) -> Self {
        MuticaType::from_type(Lazy::new(value.ty.clone()))
    }

    pub fn as_py(&self) -> PyResult<pyo3::Py<PyAny>> {
        let mut rec_detector = Vec::new();
        self.__as_py(&mut rec_detector)
    }
}

impl MuticaType {
    #[stacksafe::stacksafe]
    pub fn __as_py(
        &self,
        rec_detector: &mut Vec<(TaggedPtr<()>, pyo3::Py<PyDict>)>,
    ) -> pyo3::PyResult<pyo3::Py<PyAny>> {
        Python::attach(|py| match &self.ty {
            Type::Bound(type_bound) => match type_bound {
                TypeBound::Top => PyBool::new(py, true).into_py_any(py),
                TypeBound::Bottom => PyBool::new(py, false).into_py_any(py),
                _ => unreachable!(),
            },
            Type::Integer(_) => {
                let py_dict = PyDict::new(py);
                py_dict.set_item("kind", "Integer")?;
                py_dict.into_py_any(py)
            }
            Type::IntegerValue(integer_value) => {
                PyInt::new(py, integer_value.value()).into_py_any(py)
            }
            Type::Float(_) => {
                let py_dict = PyDict::new(py);
                py_dict.set_item("kind", "Float")?;
                py_dict.into_py_any(py)
            }
            Type::FloatValue(float_value) => PyFloat::new(py, float_value.value()).into_py_any(py),
            Type::Char(_) => {
                let py_dict = PyDict::new(py);
                py_dict.set_item("kind", "Char")?;
                py_dict.into_py_any(py)
            }
            Type::CharValue(character_value) => {
                PyString::new(py, &character_value.value().to_string()).into_py_any(py)
            }
            Type::Tuple(tuple) => PyTuple::new(
                py,
                tuple
                    .types()
                    .iter()
                    .map(|e| MuticaType::from_type(e.clone()).__as_py(rec_detector))
                    .collect::<Result<Vec<_>, _>>()?,
            )?
            .into_py_any(py),
            Type::List(list) => PyList::new(
                py,
                list.iter()
                    .map(|e| MuticaType::from_type(e.clone()).__as_py(rec_detector))
                    .collect::<Result<Vec<_>, _>>()?,
            )?
            .into_py_any(py),
            Type::Generalize(generalize) => {
                let py_dict = PyDict::new(py);
                py_dict.set_item(
                    "types",
                    generalize
                        .types()
                        .iter()
                        .map(|e| MuticaType::from_type(e.clone()).__as_py(rec_detector))
                        .collect::<Result<Vec<_>, _>>()?,
                )?;
                py_dict.set_item("kind", "Generalize")?;
                py_dict.into_py_any(py)
            }
            Type::Specialize(specialize) => {
                let py_dict = PyDict::new(py);
                py_dict.set_item(
                    "types",
                    specialize
                        .types()
                        .iter()
                        .map(|e| MuticaType::from_type(e.clone()).__as_py(rec_detector))
                        .collect::<Result<Vec<_>, _>>()?,
                )?;
                py_dict.set_item("kind", "Specialize")?;
                py_dict.into_py_any(py)
            }
            Type::FixPoint(fix_point) => match fix_point.reference().upgrade() {
                Some(inner) => {
                    let inner = match inner.as_ref().get_inner() {
                        Some(ty) => ty,
                        None => {
                            return PyNone::get(py).into_py_any(py);
                        }
                    };
                    let tagged_ptr = inner.tagged_ptr();
                    if let Some((_, py_obj)) =
                        rec_detector.iter().find(|(ptr, _)| ptr == &tagged_ptr)
                    {
                        return py_obj.into_py_any(py);
                    }

                    let py_dict = PyDict::new(py);
                    let placeholder_obj = py_dict.unbind();
                    rec_detector.push((tagged_ptr.clone(), placeholder_obj.clone_ref(py)));

                    let bound_placeholder = placeholder_obj.bind(py);
                    let ty = MuticaType::from_type(inner.clone()).__as_py(rec_detector)?;
                    rec_detector.pop();
                    bound_placeholder.set_item("kind", "FixPoint")?;
                    bound_placeholder.set_item("inner", ty)?;
                    bound_placeholder.set_item(
                        "tagged_ptr",
                        PyTuple::new(py, vec![tagged_ptr.ptr() as usize, tagged_ptr.tag()])?,
                    )?;
                    bound_placeholder.into_py_any(py)
                }
                None => PyNone::get(py).into_py_any(py),
            },
            Type::Invoke(invoke) => {
                let py_dict = PyDict::new(py);
                py_dict.set_item(
                    "function",
                    MuticaType::from_type(invoke.func().clone()).__as_py(rec_detector)?,
                )?;
                py_dict.set_item(
                    "argument",
                    MuticaType::from_type(invoke.arg().clone()).__as_py(rec_detector)?,
                )?;
                py_dict.set_item(
                    "continuation",
                    match invoke.continuation() {
                        Some(cont) => MuticaType::from_type(cont.clone()).__as_py(rec_detector)?,
                        None => py.None(),
                    },
                )?;
                py_dict.set_item(
                    "perform_handler",
                    match invoke.perform_handler() {
                        Some(handler) => {
                            MuticaType::from_type(handler.clone()).__as_py(rec_detector)?
                        }
                        None => py.None(),
                    },
                )?;
                py_dict.set_item("kind", "Invoke")?;
                py_dict.into_py_any(py)
            }
            Type::Variable(variable) => {
                let py_dict = PyDict::new(py);
                py_dict.set_item("kind", "Variable")?;
                py_dict.set_item("debruijn_index", variable.debruijn_index())?;
                py_dict.into_py_any(py)
            }
            Type::Closure(closure) => {
                let py_dict = PyDict::new(py);
                py_dict.set_item(
                    "closure_env",
                    closure
                        .env()
                        .iter()
                        .map(|e| {
                            e.iter()
                                .map(|ty| MuticaType::from_type(ty.clone()).__as_py(rec_detector))
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )?;
                py_dict.set_item(
                    "branches",
                    closure
                        .branches()
                        .iter()
                        .map(|(branch, env_idx)| {
                            let py_dict = PyDict::new(py);
                            py_dict.set_item("env_index", PyInt::new(py, *env_idx as i64))?;
                            py_dict.set_item(
                                "pattern",
                                MuticaType::from_type(branch.pattern().clone())
                                    .__as_py(rec_detector)?,
                            )?;
                            py_dict.set_item(
                                "expression",
                                MuticaType::from_type(branch.expr().clone())
                                    .__as_py(rec_detector)?,
                            )?;
                            Ok::<pyo3::Bound<'_, PyDict>, pyo3::PyErr>(py_dict)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )?;
                py_dict.set_item("kind", "Closure")?;
                py_dict.into_py_any(py)
            }
            Type::Opcode(opcode) => {
                let py_dict = PyDict::new(py);
                py_dict.set_item(
                    "opcode",
                    match opcode {
                        Opcode::Add => PyString::new(py, "Add").into_any(),
                        Opcode::Sub => PyString::new(py, "Sub").into_any(),
                        Opcode::Mul => PyString::new(py, "Mul").into_any(),
                        Opcode::Div => PyString::new(py, "Div").into_any(),
                        Opcode::Mod => PyString::new(py, "Mod").into_any(),
                        Opcode::Less => PyString::new(py, "Less").into_any(),
                        Opcode::Greater => PyString::new(py, "Greater").into_any(),
                        Opcode::Is => PyString::new(py, "Is").into_any(),
                        Opcode::Neg => PyString::new(py, "Neg").into_any(),
                        Opcode::IO(name) => PyTuple::new(py, ["IO", name.as_str()])?.into_any(),
                        Opcode::Opcode => PyString::new(py, "Opcode").into_any(),
                        _ => PyString::new(py, "Unknown").into_any(),
                    },
                )?;
                py_dict.set_item("kind", "Opcode")?;
                py_dict.into_py_any(py)
            }
            Type::Namespace(namespace) => {
                let py_dict = PyDict::new(py);
                py_dict.set_item("tag", namespace.tag())?;
                py_dict.set_item(
                    "expression",
                    MuticaType::from_type(namespace.expr().clone()).__as_py(rec_detector)?,
                )?;
                py_dict.set_item("kind", "Namespace")?;
                py_dict.into_py_any(py)
            }
            Type::Pattern(pattern) => {
                let py_dict = PyDict::new(py);
                py_dict.set_item("debruijn_index", pattern.debruijn_index())?;
                py_dict.set_item(
                    "expression",
                    MuticaType::from_type(pattern.expr().clone()).__as_py(rec_detector)?,
                )?;
                py_dict.set_item("kind", "Pattern")?;
                py_dict.into_py_any(py)
            }
            Type::Lazy(lazy) => {
                let py_dict = PyDict::new(py);
                py_dict.set_item(
                    "expression",
                    MuticaType::from_type(lazy.value().clone()).__as_py(rec_detector)?,
                )?;
                py_dict.set_item("kind", "Lazy")?;
                py_dict.into_py_any(py)
            }
            Type::Rot(rotate) => {
                let py_dict = PyDict::new(py);
                py_dict.set_item(
                    "expression",
                    MuticaType::from_type(rotate.value().clone()).__as_py(rec_detector)?,
                )?;
                py_dict.set_item("kind", "Rotate")?;
                py_dict.into_py_any(py)
            }
        })
    }
}

impl MuticaType {
    fn from_type(ty: Type<TypeGcOnceLock>) -> Self {
        MuticaType { ty }
    }
}

#[pyo3::pyclass]
pub struct MuticaGC {
    gc: GC<TypeGcOnceLock>,
}

#[pyo3::pymethods]
impl MuticaGC {
    #[new]
    pub fn new() -> Self {
        MuticaGC { gc: GC::new() }
    }

    pub fn collect(&mut self) {
        self.gc.collect();
    }
}

impl MuticaGC {
    pub fn gc(&mut self) -> &mut GC<TypeGcOnceLock> {
        &mut self.gc
    }
}

#[pyo3::pyclass]
pub struct MuticaEngine {
    scheduler: LinearScheduler<TypeGcOnceLock>,
}

type IOHandler = Box<
    dyn Fn(
            &Type<TypeGcOnceLock>,
            &Type<TypeGcOnceLock>,
        )
            -> Result<Option<Type<TypeGcOnceLock>>, TypeError<Type<TypeGcOnceLock>, TypeGcOnceLock>>
        + Send
        + Sync,
>;

impl MuticaEngine {
    pub fn new_engine(init_type: Type<TypeGcOnceLock>, io_handler: Option<IOHandler>) -> Self {
        MuticaEngine {
            scheduler: LinearScheduler::new(init_type, io_handler),
        }
    }
}

#[pyo3::pymethods]
impl MuticaEngine {
    #[new]
    pub fn new() -> Self {
        MuticaEngine {
            scheduler: LinearScheduler::new(TypeBound::top(), None),
        }
    }

    pub fn load(&mut self, expr: &str, filepath: Option<&str>, gc: &mut MuticaGC) -> Vec<String> {
        let path = if let Some(fp) = filepath {
            std::path::PathBuf::from(fp)
        } else {
            std::path::PathBuf::from("<input>")
        };
        let mut imported_ast = std::collections::HashMap::new();
        let mut cycle_detector = FastCycleDetector::new();
        let mut builder_errors = Vec::new();
        let mut error_messages = Vec::new();
        let mut multifile_builder =
            MultiFileBuilder::new(&mut imported_ast, &mut cycle_detector, &mut builder_errors);
        let (ast, source) = multifile_builder.build(path.clone(), expr.to_string());
        // 直接使用 MultiFileBuilder 构建
        let basic = match ast {
            Some(ast) if builder_errors.is_empty() => ast,
            None | Some(_) => {
                // 报告构建错误
                for error_with_loc in &builder_errors {
                    let (filepath, source_content) =
                        if let Some(location) = error_with_loc.location() {
                            let source = location.source();
                            (source.filepath(), source.content().to_string())
                        } else {
                            (path.to_string_lossy().to_string(), expr.to_string())
                        };

                    match error_with_loc.value() {
                        MultiFileBuilderError::SyntaxError(e) => {
                            let syntax_error = SyntaxError::new(e.clone());
                            let report = syntax_error.report(filepath.clone(), &source_content);
                            let mut buffer = Vec::new();
                            report
                                .write(
                                    (filepath, ariadne::Source::from(source_content)),
                                    &mut buffer,
                                )
                                .ok();
                            error_messages.push(String::from_utf8_lossy(&buffer).to_string());
                        }
                        MultiFileBuilderError::RecoveryError(e) => {
                            let report = mutica_compiler::parser::report_error_recovery(
                                e,
                                &filepath,
                                &source_content,
                            );
                            let mut buffer = Vec::new();
                            report
                                .write(
                                    (filepath.as_str(), ariadne::Source::from(source_content)),
                                    &mut buffer,
                                )
                                .ok();
                            error_messages.push(String::from_utf8_lossy(&buffer).to_string());
                        }
                        MultiFileBuilderError::IOError(e) => {
                            let range = error_with_loc
                                .location()
                                .map(|r| r.span().clone())
                                .unwrap_or(0..0);
                            let report = ariadne::Report::build(
                                ariadne::ReportKind::Error,
                                filepath.as_str(),
                                range.start,
                            )
                            .with_label(
                                ariadne::Label::new((filepath.as_str(), range)).with_message(e),
                            )
                            .finish();
                            let mut buffer = Vec::new();
                            report
                                .write(
                                    (filepath.as_str(), ariadne::Source::from(source_content)),
                                    &mut buffer,
                                )
                                .ok();
                            error_messages.push(String::from_utf8_lossy(&buffer).to_string());
                        }
                    }
                }
                return error_messages;
            }
        };

        let linearized = basic
            .0
            .linearize(&mut LinearizeContext::new(), basic.0.location())
            .finalize();
        // println!("Linearized AST: {:#?}", linearized);
        let mut flow_errors = Vec::new();
        let flowed = linearized.flow(
            &mut ParseContext::new(),
            false,
            linearized.location(),
            &mut flow_errors,
        );

        if !flow_errors.is_empty() {
            // 获取源文件信息用于错误报告
            let filepath = source.filepath();
            let source_content = source.content();
            // 报告所有错误
            let mut has_error = false;
            for e in &flow_errors {
                let filepath = e
                    .location()
                    .map(|loc| loc.source().filepath())
                    .unwrap_or_else(|| filepath.clone());
                let source_content = e
                    .location()
                    .map(|loc| loc.source().content().to_string())
                    .unwrap_or_else(|| source_content.to_string());
                let mut buffer = Vec::new();
                e.report()
                    .write(
                        (filepath, ariadne::Source::from(source_content)),
                        &mut buffer,
                    )
                    .ok();
                error_messages.push(String::from_utf8_lossy(&buffer).to_string());
                if !e.is_warning() {
                    has_error = true;
                }
            }
            if has_error {
                return error_messages;
            }
        }

        let flowed = flowed.ty().clone();
        let mut roots = RootStack::new();
        let built_type = match flowed.to_type(
            &mut BuildContext::new(),
            &mut PatternCounter::new(),
            false,
            gc.gc(),
            &mut roots,
            flowed.location(),
        ) {
            Ok(result) => result,
            Err(Ok(type_error)) => {
                error_messages.push(format!("Type building error: {:?}", type_error));
                return error_messages;
            }
            Err(Err(parse_error)) => {
                // 获取源文件信息用于错误报告
                let filepath = source.filepath();
                let source_content = source.content().to_string();
                let mut buffer = Vec::new();
                parse_error
                    .report()
                    .write(
                        (filepath, ariadne::Source::from(source_content)),
                        &mut buffer,
                    )
                    .ok();
                error_messages.push(String::from_utf8_lossy(&buffer).to_string());
                return error_messages;
            }
        };

        self.scheduler =
            roots.context(|_| scheduler::LinearScheduler::new(built_type.ty().clone(), None)); // 确保 roots 直到 linear_scheduler 被创建完成才丢弃
        Vec::new()
    }

    pub fn step(&mut self, gc: &mut MuticaGC) -> PyResult<bool> {
        self.scheduler.step(gc.gc()).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(MuticaError { err: e }.message())
        })
    }

    pub fn get_current_type(&self) -> MuticaType {
        MuticaType {
            ty: self.scheduler.current().clone(),
        }
    }
}

#[pyo3::pyclass]
pub struct MuticaError {
    err: TypeError<Type<TypeGcOnceLock>, TypeGcOnceLock>,
}

#[pyo3::pymethods]
impl MuticaError {
    #[getter]
    pub fn message(&self) -> String {
        self.err.to_string()
    }
    #[getter]
    pub fn brief(&self) -> String {
        match &self.err {
            TypeError::UnresolvableType => "Unresolvable Type".to_string(),
            TypeError::InfiniteRecursion => "Infinite Recursion".to_string(),
            TypeError::RedeclaredType => "Redeclared Type".to_string(),
            TypeError::NonApplicableType(_) => "Non-Applicable Type".to_string(),
            TypeError::TupleIndexOutOfBounds(_) => "Tuple Index Out Of Bounds".to_string(),
            TypeError::TypeMismatch(value) => format!("Type Mismatch: Expected {}", value.1),
            TypeError::UnboundVariable(_) => "Unbound Variable".to_string(),
            TypeError::AssertFailed(_) => "Assert Failed".to_string(),
            TypeError::MissingContinuation(_) => "Missing Continuation".to_string(),
            TypeError::MissingPerformHandler(_) => "Missing Perform Handler".to_string(),
            TypeError::RuntimeError(error) => format!("Runtime Error: {}", error),
            TypeError::OtherError(error) => format!("Other Error: {}", error),
            TypeError::Perform(_) => "Perform".to_string(),
            TypeError::Break(_) => "Break".to_string(),
            TypeError::Resume(_) => "Resume".to_string(),
            _ => "Unknown Error".to_string(),
        }
    }
}

#[pymodule]
#[pyo3(name = "mutica_py")]
fn mutica_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MuticaType>()?;
    m.add_class::<MuticaGC>()?;
    m.add_class::<MuticaEngine>()?;
    m.add_class::<MuticaError>()?;
    Ok(())
}
